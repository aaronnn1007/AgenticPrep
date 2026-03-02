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

export interface SubmitAnswerResponse {
  interview_id: string;
  scores: ScoresModel;
  recommendations: RecommendationsModel;
  transcript: string;
  message: string;
}
