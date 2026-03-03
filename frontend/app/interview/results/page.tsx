"use client";

import { Suspense, useState, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useInterviewStore } from "@/store/interviewStore";

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

function ScoreBar({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  const color =
    value >= 70
      ? "bg-green-500"
      : value >= 40
        ? "bg-yellow-500"
        : "bg-red-500";

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span className="text-muted-foreground">{value.toFixed(0)}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${Math.min(100, value)}%` }}
        />
      </div>
    </div>
  );
}

function CircularScore({ value }: { value: number }) {
  const pct = Math.round(value);
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;
  const color =
    pct >= 70 ? "#22c55e" : pct >= 40 ? "#eab308" : "#ef4444";

  return (
    <div className="relative flex items-center justify-center w-24 h-24">
      <svg className="w-24 h-24 -rotate-90" viewBox="0 0 100 100">
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          className="text-muted"
        />
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-700"
        />
      </svg>
      <span className="absolute text-xl font-bold">{pct}%</span>
    </div>
  );
}

function ResultsContent() {
  const router = useRouter();
  const params = useSearchParams();
  const interviewId = params.get("id") ?? "";
  const { analysisResult, question, reset } = useInterviewStore();
  const [transcriptOpen, setTranscriptOpen] = useState(false);

  // Redirect to home if no results
  useEffect(() => {
    if (!analysisResult) {
      router.replace("/");
    }
  }, [analysisResult, router]);

  if (!analysisResult) {
    return null; // Will redirect
  }

  const { scores, recommendations, transcript } = analysisResult;

  function handleNewInterview() {
    reset();
    window.location.href = "/";
  }

  return (
    <main className="min-h-screen bg-background p-4 text-foreground md:p-8">
      <div className="mx-auto max-w-3xl space-y-6">
        {/* ── Header ── */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Interview Results</h1>
            <p className="text-sm text-muted-foreground">
              Session {interviewId}
            </p>
          </div>
          <CircularScore value={scores.overall} />
        </div>

        {/* ── Scores ── */}
        <Card className="border-border/60 shadow-lg">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Performance Scores</CardTitle>
            <CardDescription>
              Breakdown across technical, communication, and behavioral
              dimensions.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <ScoreBar label="Technical" value={scores.technical} />
            <ScoreBar label="Communication" value={scores.communication} />
            <ScoreBar label="Behavioral" value={scores.behavioral} />
            <div className="border-t border-border pt-3">
              <ScoreBar label="Overall" value={scores.overall} />
            </div>
          </CardContent>
        </Card>

        {/* ── Recommendations ── */}
        <div className="grid gap-6 sm:grid-cols-2">
          {/* Strengths */}
          <Card className="border-border/60">
            <CardHeader className="pb-2">
              <CardTitle className="text-base text-green-600 dark:text-green-400">
                ✅ Strengths
              </CardTitle>
            </CardHeader>
            <CardContent>
              {recommendations.strengths.length > 0 ? (
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {recommendations.strengths.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">—</p>
              )}
            </CardContent>
          </Card>

          {/* Weaknesses */}
          <Card className="border-border/60">
            <CardHeader className="pb-2">
              <CardTitle className="text-base text-red-600 dark:text-red-400">
                ⚠️ Areas to Improve
              </CardTitle>
            </CardHeader>
            <CardContent>
              {recommendations.weaknesses.length > 0 ? (
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {recommendations.weaknesses.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">—</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* ── Improvement Plan ── */}
        {recommendations.improvement_plan.length > 0 && (
          <Card className="border-border/60">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">📋 Improvement Plan</CardTitle>
            </CardHeader>
            <CardContent>
              <ol className="list-decimal space-y-1 pl-5 text-sm">
                {recommendations.improvement_plan.map((step, i) => (
                  <li key={i}>{step}</li>
                ))}
              </ol>
            </CardContent>
          </Card>
        )}

        {/* ── Transcript (collapsible) ── */}
        <Card className="border-border/60">
          <CardHeader
            className="pb-2 cursor-pointer select-none"
            onClick={() => setTranscriptOpen((o) => !o)}
          >
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">🗣️ Your Answer (Transcript)</CardTitle>
              <span className="text-sm text-muted-foreground">
                {transcriptOpen ? "▲ Hide" : "▼ Show"}
              </span>
            </div>
            <CardDescription>
              {question && `Question: ${question}`}
            </CardDescription>
          </CardHeader>
          {transcriptOpen && (
            <CardContent>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground">
                {transcript || "No transcript available."}
              </p>
            </CardContent>
          )}
        </Card>

        {/* ── Actions ── */}
        <div className="flex justify-center pb-8">
          <Button size="lg" onClick={handleNewInterview}>
            Start New Interview
          </Button>
        </div>
      </div>
    </main>
  );
}

export default function ResultsPage() {
  return (
    <Suspense
      fallback={
        <main className="flex min-h-screen items-center justify-center bg-background text-foreground">
          <Spinner className="h-8 w-8 text-muted-foreground" />
        </main>
      }
    >
      <ResultsContent />
    </Suspense>
  );
}
