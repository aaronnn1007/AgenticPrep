"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { useInterviewStore } from "@/store/interviewStore";
import type { Role, ExperienceLevel, StartInterviewResponse } from "@/types/interview";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";

const ROLES = [
  { value: "frontend", label: "Frontend Engineer" },
  { value: "backend", label: "Backend Engineer" },
  { value: "fullstack", label: "Fullstack Engineer" },
  { value: "data_scientist", label: "Data Scientist" },
] as const;

const EXPERIENCE_LEVELS = [
  { value: "junior", label: "Junior", years: "0–2 yrs" },
  { value: "mid", label: "Mid-level", years: "2–5 yrs" },
  { value: "senior", label: "Senior", years: "5+ yrs" },
] as const;

const FEATURES = [
  {
    icon: "🎯",
    title: "Technical Accuracy",
    description: "Evaluated against real-world engineering standards.",
  },
  {
    icon: "🗣️",
    title: "Communication Score",
    description: "Clarity, structure, and conciseness of your answers.",
  },
  {
    icon: "👁️",
    title: "Body Language Analysis",
    description: "Eye contact, posture, and non-verbal confidence signals.",
  },
  {
    icon: "📊",
    title: "Actionable Feedback",
    description: "Personalised coaching report after every session.",
  },
];

export default function LandingPage() {
  const router = useRouter();
  const [role, setRole] = useState<string>("");
  const [experience, setExperience] = useState<string>("");
  const [numQuestions, setNumQuestions] = useState<number>(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canStart = role !== "" && experience !== "";
  const store = useInterviewStore();

  async function handleStart() {
    if (!canStart) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/start-interview`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            role,
            experience_level: experience,
            num_questions: numQuestions,
            time_per_question: 120,
          }),
        }
      );

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}`);
      }

      const data: StartInterviewResponse = await res.json();
      store.setRole(role as Role);
      store.setExperienceLevel(experience as ExperienceLevel);
      store.setSession(
        data.interview_id,
        data.question.text,
        data.num_questions,
        data.time_per_question,
        data.question.topic,
        data.question.difficulty,
        data.question.intent,
      );

      router.push(`/interview/new?role=${encodeURIComponent(role)}&level=${encodeURIComponent(experience)}`);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to start session. Is the backend running?"
      );
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      {/* ── Nav ── */}
      <nav className="border-b border-border/60 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Image
              src="/globe.svg"
              alt="AgenticPrep Logo"
              width={24}
              height={24}
              className="-ml-0.5"
            />
            <span className="text-xl font-bold tracking-tight">AgenticPrep</span>
            <Badge variant="secondary" className="text-xs">
              Beta
            </Badge>
          </div>
          <div className="hidden items-center gap-6 text-sm text-muted-foreground sm:flex">
            <span className="cursor-pointer hover:text-foreground transition-colors">
              How it works
            </span>
            <span className="cursor-pointer hover:text-foreground transition-colors">
              Pricing
            </span>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="mx-auto max-w-6xl px-6 pt-12 pb-12 text-center">
        <Badge variant="outline" className="mb-6 gap-1.5 px-3 py-1 text-xs font-medium">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-500" />
          AI-Powered Interview Coach
        </Badge>

        <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl lg:text-6xl">
          Ace your next{" "}
          <span className="bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            technical interview
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground">
          Practice technical interviews with AI-powered analysis of your{" "}
          <strong className="text-foreground">technical accuracy</strong>,{" "}
          <strong className="text-foreground">communication</strong>, and{" "}
          <strong className="text-foreground">body language</strong> — all in one
          session.
        </p>
      </section>

      {/* ── Session Setup Card ── */}
      <section className="mx-auto max-w-lg px-6 pb-20">
        <Card className="shadow-xl border-border/60">
          <CardHeader className="pb-4">
            <CardTitle className="text-xl">Configure your session</CardTitle>
            <CardDescription>
              Select a role and experience level to receive targeted questions.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-5">
            {/* Role picker */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Role</label>
              <Select value={role} onValueChange={setRole}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Choose a role…" />
                </SelectTrigger>
                <SelectContent>
                  {ROLES.map((r) => (
                    <SelectItem key={r.value} value={r.value}>
                      {r.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Experience level toggle */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Experience level</label>
              <div className="grid grid-cols-3 gap-2">
                {EXPERIENCE_LEVELS.map((lvl) => (
                  <button
                    key={lvl.value}
                    type="button"
                    onClick={() => setExperience(lvl.value)}
                    className={[
                      "flex flex-col items-center justify-center rounded-lg border px-2 py-3 text-sm transition-all",
                      "hover:border-primary/60 hover:bg-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                      experience === lvl.value
                        ? "border-primary bg-primary/10 font-semibold text-primary"
                        : "border-border text-muted-foreground",
                    ].join(" ")}
                  >
                    <span className="font-medium">{lvl.label}</span>
                    <span className="text-xs opacity-70">{lvl.years}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Number of questions */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Number of questions</label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min={3}
                  max={10}
                  value={numQuestions}
                  onChange={(e) => setNumQuestions(Number(e.target.value))}
                  className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-muted accent-primary"
                />
                <span className="w-8 text-center text-sm font-semibold tabular-nums">
                  {numQuestions}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                {numQuestions} questions &middot; ~{numQuestions * 2} min
              </p>
            </div>

            {/* Error banner */}
            {error && (
              <p className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {error}
              </p>
            )}

            {/* CTA */}
            <Button
              className="w-full"
              size="lg"
              disabled={!canStart || loading}
              onClick={handleStart}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg
                    className="h-4 w-4 animate-spin"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                  </svg>
                  Starting session…
                </span>
              ) : (
                "Start Interview →"
              )}
            </Button>

            <p className="text-center text-xs text-muted-foreground">
              Sessions are recorded locally and never shared without your consent.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* ── Feature Grid ── */}
      <section className="border-t border-border/60 bg-muted/30 px-6 py-20">
        <div className="mx-auto max-w-6xl">
          <h2 className="mb-2 text-center text-2xl font-bold">
            Every dimension of your performance, analysed
          </h2>
          <p className="mb-12 text-center text-muted-foreground">
            AgenticPrep uses specialised AI agents working in parallel — so you
            get fast, deep insights.
          </p>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {FEATURES.map((f) => (
              <Card
                key={f.title}
                className="border-border/60 bg-background/80 transition-shadow hover:shadow-md"
              >
                <CardHeader className="pb-2">
                  <div className="mb-2 text-3xl">{f.icon}</div>
                  <CardTitle className="text-base">{f.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-sm leading-relaxed">
                    {f.description}
                  </CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-border/60 px-6 py-8 text-center text-sm text-muted-foreground">
        © {new Date().getFullYear()} AgenticPrep — Built with Next.js 15 &amp; FastAPI
      </footer>
    </main>
  );
}
