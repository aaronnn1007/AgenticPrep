import { create } from "zustand";
import type {
  Role,
  ExperienceLevel,
  RecordingState,
  SubmitAnswerResponse,
} from "@/types/interview";

export type { RecordingState };

const API = () => process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface InterviewState {
  /* session config (set from landing page) */
  role: Role | "";
  experienceLevel: ExperienceLevel | "";

  /* session data (set after /start-interview response) */
  interviewId: string | null;
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
  analysisResult: SubmitAnswerResponse | null;

  /* media */
  stream: MediaStream | null;

  /* actions */
  setRole: (role: Role | "") => void;
  setExperienceLevel: (level: ExperienceLevel | "") => void;
  setSession: (id: string, question: string, topic?: string, difficulty?: number, intent?: string[]) => void;
  setFetchingQuestion: (v: boolean) => void;
  setFetchError: (err: string | null) => void;
  setRecordingState: (state: RecordingState) => void;
  setStream: (stream: MediaStream | null) => void;
  startRecording: () => void;
  stopRecording: () => void;
  uploadAnswer: (blob: Blob) => Promise<void>;
  reset: () => void;
}

const initialState = {
  role: "" as Role | "",
  experienceLevel: "" as ExperienceLevel | "",
  interviewId: null as string | null,
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
  analysisResult: null as SubmitAnswerResponse | null,
  stream: null as MediaStream | null,
};

export const useInterviewStore = create<InterviewState>((set, get) => ({
  ...initialState,

  setRole: (role) => set({ role }),
  setExperienceLevel: (level) => set({ experienceLevel: level }),
  setSession: (id, question, topic, difficulty, intent) =>
    set({
      interviewId: id,
      question,
      questionTopic: topic ?? null,
      questionDifficulty: difficulty ?? null,
      questionIntent: intent ?? null,
      fetchingQuestion: false,
      fetchError: null,
    }),
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
      get().uploadAnswer(blob);
    };

    recorder.start(1000); // collect in 1-second chunks
    set({ mediaRecorder: recorder, chunks: [], recordingState: "recording", uploadError: null });
  },

  stopRecording: () => {
    const { mediaRecorder } = get();
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop(); // triggers onstop → uploadAnswer
    }
  },

  /* ─── Upload ─── */
  uploadAnswer: async (blob: Blob) => {
    const { interviewId, role, experienceLevel } = get();
    set({ uploadError: null });

    try {
      const formData = new FormData();
      formData.append("interview_id", interviewId ?? "");
      formData.append("role", role || "fullstack");
      formData.append("experience_level", experienceLevel || "mid");
      formData.append("audio_file", blob, `answer_${interviewId}.webm`);
      // video_file is optional — the webm already contains both tracks

      const res = await fetch(`${API()}/submit-answer`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(detail.detail ?? `Server responded with ${res.status}`);
      }

      const result: SubmitAnswerResponse = await res.json();
      set({ analysisResult: result, recordingState: "idle" });
    } catch (err) {
      set({
        uploadError: err instanceof Error ? err.message : "Upload failed.",
        recordingState: "idle",
      });
    }
  },

  reset: () => set(initialState),
}));
