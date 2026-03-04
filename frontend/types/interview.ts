export type Role =
  | "frontend"
  | "backend"
  | "fullstack"
  | "data_scientist";

export type ExperienceLevel = "junior" | "mid" | "senior";

export type RecordingState = "idle" | "recording" | "processing";

export interface QuestionModel {
  text: string;
  topic: string;
  difficulty: number;
  intent: string[];
}

export interface StartInterviewResponse {
  interview_id: string;
  question: QuestionModel;
  num_questions: number;
  time_per_question: number;
  current_question_index: number;
  message: string;
}

export interface NextQuestionResponse {
  interview_id: string;
  question: QuestionModel;
  current_question_index: number;
  is_last: boolean;
  message: string;
}

export interface SubmitQuestionAnswerResponse {
  interview_id: string;
  question_index: number;
  status: string;
  message: string;
}

export interface ScoresModel {
  technical: number;
  communication: number;
  behavioral: number;
  overall: number;
}

export interface RecommendationsModel {
  strengths: string[];
  weaknesses: string[];
  improvement_plan: string[];
}

export interface QuestionResultModel {
  question: QuestionModel | null;
  transcript: string;
  scores: ScoresModel;
}

export interface CompleteInterviewResponse {
  interview_id: string;
  question_results: QuestionResultModel[];
  aggregate_scores: ScoresModel;
  recommendations: RecommendationsModel;
  message: string;
}

/** Legacy single-question response (kept for backward compat) */
export interface SubmitAnswerResponse {
  interview_id: string;
  scores: ScoresModel;
  recommendations: RecommendationsModel;
  transcript: string;
  message: string;
}
