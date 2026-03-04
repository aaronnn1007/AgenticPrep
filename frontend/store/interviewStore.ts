import { create } from "zustand";
import type {
  Role,
  ExperienceLevel,
  RecordingState,
  QuestionModel,
  CompleteInterviewResponse,
  SubmitAnswerResponse,
} from "@/types/interview";
import {
  submitAnswer,
  submitQuestionAnswer,
  fetchNextQuestion,
  completeInterview,
  ApiError,
} from "@/lib/api";

export type { RecordingState };

interface InterviewState {
  /* session config (set from landing page) */
  role: Role | "";
  experienceLevel: ExperienceLevel | "";

  /* session data (set after /start-interview response) */
  interviewId: string | null;
  numQuestions: number;
  timePerQuestion: number; // seconds
  currentQuestionIndex: number;
  questions: QuestionModel[];

  /* current question shortcut */
  question: string | null;
  questionTopic: string | null;
  questionDifficulty: number | null;
  questionIntent: string[] | null;

  /* fetch flags */
  fetchingQuestion: boolean;
  fetchError: string | null;

  /* recording */
  recordingState: RecordingState;
  mediaRecorder: MediaRecorder | null;
  chunks: Blob[];

  /* upload */
  uploadError: string | null;

  /* multi-question completion result */
  completionResult: CompleteInterviewResponse | null;
  completing: boolean;

  /* legacy single-question result */
  analysisResult: SubmitAnswerResponse | null;

  /* timer */
  timeRemaining: number; // seconds
  timerInterval: ReturnType<typeof setInterval> | null;

  /* media */
  stream: MediaStream | null;

  /* actions */
  setRole: (role: Role | "") => void;
  setExperienceLevel: (level: ExperienceLevel | "") => void;
  setSession: (
    id: string,
    question: string,
    numQuestions: number,
    timePerQuestion: number,
    topic?: string,
    difficulty?: number,
    intent?: string[],
  ) => void;
  setFetchingQuestion: (v: boolean) => void;
  setFetchError: (err: string | null) => void;
  setRecordingState: (state: RecordingState) => void;
  setStream: (stream: MediaStream | null) => void;

  /* recording */
  startRecording: () => void;
  stopRecording: () => void;

  /* multi-question flow */
  uploadQuestionAnswer: (blob: Blob) => Promise<void>;
  advanceToNextQuestion: () => Promise<void>;
  finishInterview: () => Promise<void>;

  /* legacy */
  uploadAnswer: (blob: Blob) => Promise<void>;

  /* timer */
  startTimer: () => void;
  stopTimer: () => void;
  tickTimer: () => void;

  reset: () => void;
}

const initialState = {
  role: "" as Role | "",
  experienceLevel: "" as ExperienceLevel | "",
  interviewId: null as string | null,
  numQuestions: 5,
  timePerQuestion: 120,
  currentQuestionIndex: 0,
  questions: [] as QuestionModel[],
  question: null as string | null,
  questionTopic: null as string | null,
  questionDifficulty: null as number | null,
  questionIntent: null as string[] | null,
  fetchingQuestion: false,
  fetchError: null as string | null,
  recordingState: "idle" as RecordingState,
  mediaRecorder: null as MediaRecorder | null,
  chunks: [] as Blob[],
  uploadError: null as string | null,
  completionResult: null as CompleteInterviewResponse | null,
  completing: false,
  analysisResult: null as SubmitAnswerResponse | null,
  timeRemaining: 120,
  timerInterval: null as ReturnType<typeof setInterval> | null,
  stream: null as MediaStream | null,
};

export const useInterviewStore = create<InterviewState>((set, get) => ({
  ...initialState,

  setRole: (role) => set({ role }),
  setExperienceLevel: (level) => set({ experienceLevel: level }),

  setSession: (id, question, numQuestions, timePerQuestion, topic, difficulty, intent) => {
    const qModel: QuestionModel = {
      text: question,
      topic: topic ?? "",
      difficulty: difficulty ?? 0.5,
      intent: intent ?? [],
    };
    set({
      interviewId: id,
      question,
      questionTopic: topic ?? null,
      questionDifficulty: difficulty ?? null,
      questionIntent: intent ?? null,
      numQuestions,
      timePerQuestion,
      timeRemaining: timePerQuestion,
      currentQuestionIndex: 0,
      questions: [qModel],
      fetchingQuestion: false,
      fetchError: null,
    });
  },

  setFetchingQuestion: (v) => set({ fetchingQuestion: v }),
  setFetchError: (err) => set({ fetchError: err, fetchingQuestion: false }),
  setRecordingState: (recordingState) => set({ recordingState }),
  setStream: (stream) => set({ stream }),

  /* ─── Recording ─── */
  startRecording: () => {
    const { stream } = get();
    if (!stream) return;

    const chunks: Blob[] = [];
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")
      ? "video/webm;codecs=vp9,opus"
      : "video/webm";
    const recorder = new MediaRecorder(stream, { mimeType });

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType });
      set({ chunks, recordingState: "processing" });
      get().uploadQuestionAnswer(blob);
    };

    recorder.start(1000);
    set({
      mediaRecorder: recorder,
      chunks: [],
      recordingState: "recording",
      uploadError: null,
    });

    // Start countdown timer
    get().startTimer();
  },

  stopRecording: () => {
    const { mediaRecorder } = get();
    get().stopTimer();
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop(); // triggers onstop → uploadQuestionAnswer
    }
  },

  /* ─── Upload per-question answer ─── */
  uploadQuestionAnswer: async (blob: Blob) => {
    const { interviewId, currentQuestionIndex } = get();
    set({ uploadError: null });

    try {
      await submitQuestionAnswer({
        interview_id: interviewId ?? "",
        question_index: currentQuestionIndex,
        audio_file: blob,
        audio_filename: `answer_${interviewId}_q${currentQuestionIndex}.webm`,
      });
      set({ recordingState: "idle" });
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.detail
          : err instanceof Error
            ? err.message
            : "Upload failed.";
      set({ uploadError: message, recordingState: "idle" });
    }
  },

  /* ─── Advance to next question ─── */
  advanceToNextQuestion: async () => {
    const { interviewId } = get();
    if (!interviewId) return;

    set({ fetchingQuestion: true, fetchError: null });

    try {
      const data = await fetchNextQuestion(interviewId);
      const q = data.question;
      set((s) => ({
        currentQuestionIndex: data.current_question_index,
        question: q.text,
        questionTopic: q.topic,
        questionDifficulty: q.difficulty,
        questionIntent: q.intent,
        questions: [...s.questions, q],
        fetchingQuestion: false,
        timeRemaining: s.timePerQuestion,
        recordingState: "idle",
        uploadError: null,
        chunks: [],
      }));
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.detail
          : err instanceof Error
            ? err.message
            : "Failed to load next question.";
      set({ fetchError: message, fetchingQuestion: false });
    }
  },

  /* ─── Finish interview (run analysis) ─── */
  finishInterview: async () => {
    const { interviewId, stream } = get();
    if (!interviewId) return;

    // Kill camera immediately so the light turns off
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
    set({ completing: true, uploadError: null, stream: null });

    try {
      const result = await completeInterview(interviewId);
      set({ completionResult: result, completing: false });
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.detail
          : err instanceof Error
            ? err.message
            : "Analysis failed.";
      set({ uploadError: message, completing: false });
    }
  },

  /* ─── Legacy upload (single-question) ─── */
  uploadAnswer: async (blob: Blob) => {
    const { interviewId, role, experienceLevel } = get();
    set({ uploadError: null });

    try {
      const result = await submitAnswer({
        interview_id: interviewId ?? "",
        role: role || "fullstack",
        experience_level: experienceLevel || "mid",
        audio_file: blob,
        audio_filename: `answer_${interviewId}.webm`,
      });
      set({ analysisResult: result, recordingState: "idle" });
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.detail
          : err instanceof Error
            ? err.message
            : "Upload failed.";
      set({ uploadError: message, recordingState: "idle" });
    }
  },

  /* ─── Timer ─── */
  startTimer: () => {
    const existing = get().timerInterval;
    if (existing) clearInterval(existing);

    const interval = setInterval(() => {
      get().tickTimer();
    }, 1000);
    set({ timerInterval: interval });
  },

  stopTimer: () => {
    const { timerInterval } = get();
    if (timerInterval) {
      clearInterval(timerInterval);
      set({ timerInterval: null });
    }
  },

  tickTimer: () => {
    const { timeRemaining } = get();
    if (timeRemaining <= 1) {
      // Time's up — auto-stop recording
      get().stopTimer();
      const { mediaRecorder } = get();
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
      set({ timeRemaining: 0 });
    } else {
      set({ timeRemaining: timeRemaining - 1 });
    }
  },

  reset: () => {
    const { timerInterval } = get();
    if (timerInterval) clearInterval(timerInterval);
    set({ ...initialState });
  },
}));
