"use client";

import { Suspense, useCallback, useEffect, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useInterviewStore } from "@/store/interviewStore";
import type { Role, ExperienceLevel } from "@/types/interview";

/* ------------------------------------------------------------------ */
/*  Spinner                                                           */
/* ------------------------------------------------------------------ */
function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg
      className={`animate-spin ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Inner component (reads searchParams → needs Suspense boundary)    */
/* ------------------------------------------------------------------ */
function InterviewSession() {
  const params = useSearchParams();
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);

  const {
    question,
    fetchingQuestion,
    fetchError,
    recordingState,
    stream,
    uploadError,
    analysisResult,
    interviewId,
    setRole,
    setExperienceLevel,
    setSession,
    setFetchingQuestion,
    setFetchError,
    startRecording,
    stopRecording,
    setStream,
  } = useInterviewStore();

  /* ---------- read query params once ---------- */
  const role = params.get("role") ?? "";
  const level = params.get("level") ?? "";

  useEffect(() => {
    if (role) setRole(role as Role);
    if (level) setExperienceLevel(level as ExperienceLevel);
  }, [role, level, setRole, setExperienceLevel]);

  /* ---------- fetch question on mount ---------- */
  const fetchQuestion = useCallback(async () => {
    setFetchingQuestion(true);
    setFetchError(null);
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/start-interview`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            role: role || "fullstack",
            experience_level: level || "mid",
          }),
        }
      );
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      const q = data.question;
      setSession(
        data.interview_id ?? crypto.randomUUID(),
        typeof q === "string" ? q : q?.text ?? "",
        typeof q === "object" ? q?.topic : undefined,
        typeof q === "object" ? q?.difficulty : undefined,
        typeof q === "object" ? q?.intent : undefined,
      );
    } catch (err) {
      setFetchError(
        err instanceof Error ? err.message : "Failed to load question."
      );
    }
  }, [role, level, setFetchingQuestion, setFetchError, setSession]);

  /* skip re-fetch if the landing page already saved the question to the store */
  useEffect(() => {
    if (!question) fetchQuestion();
  }, [fetchQuestion, question]);

  /* ---------- camera stream ---------- */
  useEffect(() => {
    let cancelled = false;

    async function startCamera() {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });
        if (!cancelled) {
          setStream(mediaStream);
          if (videoRef.current) {
            videoRef.current.srcObject = mediaStream;
          }
        }
      } catch {
        /* user denied — preview stays black */
      }
    }

    startCamera();

    return () => {
      cancelled = true;
      /* clean up tracks on unmount */
      useInterviewStore.getState().stream?.getTracks().forEach((t) => t.stop());
      setStream(null);
    };
  }, [setStream]);

  /* keep <video> srcObject in sync when stream changes */
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  /* ---------- navigate to results when analysis completes ---------- */
  useEffect(() => {
    if (analysisResult) {
      router.push(
        `/interview/results?id=${encodeURIComponent(interviewId ?? "")}`
      );
    }
  }, [analysisResult, interviewId, router]);

  /* ---------- render ---------- */
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 bg-background p-4 text-foreground md:p-8">
      <Card className="w-full max-w-3xl shadow-xl border-border/60">
        {/* ── Header ── */}
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-xl">Interview Session</CardTitle>
              <CardDescription className="mt-1">
                {role && level
                  ? `${role.replace("_", " ")} · ${level}`
                  : "Loading session…"}
              </CardDescription>
            </div>

            {recordingState === "recording" && (
              <Badge
                variant="destructive"
                className="flex items-center gap-1.5"
              >
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                  <span className="inline-flex h-2 w-2 rounded-full bg-red-500" />
                </span>
                REC
              </Badge>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* ── Video Preview ── */}
          <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-border bg-black">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="h-full w-full object-cover"
            />

            {/* camera icon overlay — shown until stream is ready */}
            {!stream && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-12 w-12 opacity-40"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9A2.25 2.25 0 0 0 13.5 5.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"
                  />
                </svg>
                <span className="text-sm">Requesting camera access…</span>
              </div>
            )}
          </div>

          {/* ── Question ── */}
          <div className="rounded-lg border border-border bg-muted/40 p-4">
            {fetchingQuestion ? (
              <div className="flex items-center justify-center gap-2 py-4 text-muted-foreground">
                <Spinner className="h-5 w-5" />
                <span className="text-sm">Loading question…</span>
              </div>
            ) : fetchError ? (
              <div className="space-y-3 text-center">
                <p className="text-sm text-destructive">{fetchError}</p>
                <Button variant="outline" size="sm" onClick={fetchQuestion}>
                  Retry
                </Button>
              </div>
            ) : (
              <>
                <p className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Question
                </p>
                <p className="text-base leading-relaxed">{question}</p>
              </>
            )}
          </div>

          {/* ── Controls ── */}
          {uploadError && (
            <p className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {uploadError}
            </p>
          )}

          <Button
            size="lg"
            className="w-full text-base"
            disabled={
              fetchingQuestion ||
              !!fetchError ||
              recordingState === "processing" ||
              !stream
            }
            onClick={
              recordingState === "recording" ? stopRecording : startRecording
            }
          >
            {recordingState === "processing" ? (
              <span className="flex items-center gap-2">
                <Spinner className="h-4 w-4" />
                Processing answer…
              </span>
            ) : recordingState === "recording" ? (
              <span className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-white" />
                Stop Recording
              </span>
            ) : (
              "🎤 Start Recording"
            )}
          </Button>
        </CardContent>
      </Card>
    </main>
  );
}

/* ------------------------------------------------------------------ */
/*  Page wrapper — Suspense boundary for useSearchParams               */
/* ------------------------------------------------------------------ */
export default function NewInterviewPage() {
  return (
    <Suspense
      fallback={
        <main className="flex min-h-screen items-center justify-center bg-background text-foreground">
          <Spinner className="h-8 w-8 text-muted-foreground" />
        </main>
      }
    >
      <InterviewSession />
    </Suspense>
  );
}
