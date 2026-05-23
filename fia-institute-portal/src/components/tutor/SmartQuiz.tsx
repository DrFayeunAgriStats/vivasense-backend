import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Check, X, RotateCcw, Timer, Sparkles, Trophy, Shuffle, Lightbulb,
} from "lucide-react";
import {
  pickQuestions, TOPIC_LABELS,
  type Question, type Topic, type Difficulty,
} from "@/data/plantImprovementQuestions";

const ALL_TOPICS = Object.keys(TOPIC_LABELS) as Topic[];

interface Props {
  onAnswered?: (answered: number, total: number) => void;
}

export function SmartQuiz({ onAnswered }: Props) {
  const [topics, setTopics] = useState<Topic[]>(ALL_TOPICS);
  const [difficulty, setDifficulty] = useState<Difficulty | "mixed">("mixed");
  const [count, setCount] = useState(8);
  const [timed, setTimed] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [questions, setQuestions] = useState<Question[]>(() =>
    pickQuestions({ topics: ALL_TOPICS, difficulty: "mixed", count: 8 })
  );
  const [answers, setAnswers] = useState<Record<string, number | string>>({});
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const [fillInputs, setFillInputs] = useState<Record<string, string>>({});
  const [hintShown, setHintShown] = useState<Record<string, boolean>>({});

  const score = useMemo(
    () => questions.reduce((s, q) => (isCorrect(q, answers[q.id]) ? s + 1 : s), 0),
    [questions, answers]
  );
  const answeredCount = Object.keys(revealed).length;
  const progress = (answeredCount / questions.length) * 100;
  const finished = answeredCount === questions.length && questions.length > 0;

  // Timer
  useEffect(() => {
    if (!timed || finished) return;
    const t = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [timed, finished]);

  useEffect(() => {
    onAnswered?.(answeredCount, questions.length);
  }, [answeredCount, questions.length, onAnswered]);

  const regenerate = () => {
    setQuestions(pickQuestions({ topics, difficulty, count }));
    setAnswers({});
    setRevealed({});
    setFillInputs({});
    setHintShown({});
    setSeconds(0);
  };

  const toggleTopic = (t: Topic) => {
    setTopics((curr) =>
      curr.includes(t) ? curr.filter((x) => x !== t) : [...curr, t]
    );
  };

  const submitFill = (q: Question) => {
    const val = (fillInputs[q.id] ?? "").trim();
    if (!val) return;
    setAnswers((a) => ({ ...a, [q.id]: val }));
    setRevealed((r) => ({ ...r, [q.id]: true }));
  };

  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");

  return (
    <div className="space-y-6">
      {/* Controls */}
      <Card className="p-4 sm:p-6 border-emerald-100 bg-gradient-to-br from-emerald-50/40 to-white">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 items-end">
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-emerald-800 mb-2 block">
              Difficulty
            </label>
            <div className="flex flex-wrap gap-1.5">
              {(["mixed", "easy", "medium", "hard"] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => setDifficulty(d)}
                  className={`text-xs px-3 py-1.5 rounded-full border transition ${
                    difficulty === d
                      ? "bg-emerald-700 text-white border-emerald-700"
                      : "bg-white border-emerald-200 hover:bg-emerald-50"
                  }`}
                >
                  {d.charAt(0).toUpperCase() + d.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-emerald-800 mb-2 block">
              Number of Questions
            </label>
            <div className="flex gap-1.5">
              {[5, 8, 12, 20].map((n) => (
                <button
                  key={n}
                  onClick={() => setCount(n)}
                  className={`text-xs px-3 py-1.5 rounded-full border transition ${
                    count === n
                      ? "bg-emerald-700 text-white border-emerald-700"
                      : "bg-white border-emerald-200 hover:bg-emerald-50"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setTimed((v) => !v)}
              className={`flex items-center gap-2 text-xs px-3 py-2 rounded-full border transition ${
                timed
                  ? "bg-amber-100 border-amber-300 text-amber-900"
                  : "bg-white border-emerald-200 hover:bg-emerald-50"
              }`}
            >
              <Timer className="w-3.5 h-3.5" />
              {timed ? `Timed ${mm}:${ss}` : "Enable Timer"}
            </button>
          </div>

          <Button onClick={regenerate} className="bg-emerald-700 hover:bg-emerald-800">
            <Shuffle className="w-4 h-4 mr-1.5" /> New Quiz
          </Button>
        </div>

        <div className="mt-4 pt-4 border-t border-emerald-100">
          <label className="text-xs font-semibold uppercase tracking-wider text-emerald-800 mb-2 block">
            Topics
          </label>
          <div className="flex flex-wrap gap-1.5">
            {ALL_TOPICS.map((t) => {
              const on = topics.includes(t);
              return (
                <button
                  key={t}
                  onClick={() => toggleTopic(t)}
                  className={`text-xs px-3 py-1.5 rounded-full border transition ${
                    on
                      ? "bg-emerald-100 border-emerald-400 text-emerald-900"
                      : "bg-white border-border text-muted-foreground hover:border-emerald-300"
                  }`}
                >
                  {TOPIC_LABELS[t]}
                </button>
              );
            })}
          </div>
        </div>
      </Card>

      {/* Progress / Score */}
      <Card className="p-4 sm:p-5 border-emerald-100">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2 text-sm">
            <Sparkles className="w-4 h-4 text-emerald-600" />
            <span className="font-semibold">
              {answeredCount} of {questions.length} answered
            </span>
          </div>
          <Badge className="bg-emerald-100 text-emerald-800 hover:bg-emerald-100 text-sm px-3 py-1">
            <Trophy className="w-3.5 h-3.5 mr-1" /> Score: {score} / {questions.length}
          </Badge>
        </div>
        <Progress value={progress} className="h-2" />
      </Card>

      {/* Questions */}
      <div className="space-y-4">
        {questions.map((q, qi) => {
          const show = revealed[q.id];
          const correct = isCorrect(q, answers[q.id]);
          return (
            <Card key={q.id} className="p-5 sm:p-6 border-emerald-100">
              <div className="flex items-start justify-between gap-3 mb-3 flex-wrap">
                <p className="font-semibold flex-1 min-w-0">
                  <span className="text-emerald-700 mr-2">Q{qi + 1}.</span>
                  {q.q}
                </p>
                <div className="flex gap-1.5 flex-shrink-0">
                  <Badge variant="outline" className="text-[10px] capitalize">
                    {q.type === "tf" ? "T/F" : q.type}
                  </Badge>
                  <Badge
                    variant="outline"
                    className={`text-[10px] capitalize ${
                      q.difficulty === "easy"
                        ? "border-emerald-300 text-emerald-700"
                        : q.difficulty === "medium"
                        ? "border-amber-300 text-amber-700"
                        : "border-red-300 text-red-700"
                    }`}
                  >
                    {q.difficulty}
                  </Badge>
                  <Badge variant="outline" className="text-[10px] hidden sm:inline-flex">
                    {TOPIC_LABELS[q.topic]}
                  </Badge>
                </div>
              </div>

              {/* MCQ / TF / Scenario */}
              {(q.type === "mcq" || q.type === "tf" || q.type === "scenario") && q.options && (
                <div className={`grid gap-2 ${q.type === "tf" ? "sm:grid-cols-2" : "sm:grid-cols-2"}`}>
                  {q.options.map((opt, oi) => {
                    const chosen = answers[q.id] === oi;
                    const isAns = q.answer === oi;
                    return (
                      <button
                        key={oi}
                        disabled={show}
                        onClick={() => {
                          setAnswers((a) => ({ ...a, [q.id]: oi }));
                          setRevealed((r) => ({ ...r, [q.id]: true }));
                        }}
                        className={`text-left px-4 py-3 rounded-lg border text-sm transition-all flex items-center justify-between gap-2 ${
                          show && isAns
                            ? "bg-emerald-50 border-emerald-500 text-emerald-900"
                            : show && chosen && !isAns
                            ? "bg-red-50 border-red-300 text-red-800"
                            : "border-border hover:border-emerald-300 hover:bg-emerald-50/50"
                        }`}
                      >
                        <span>{opt}</span>
                        {show && isAns && <Check className="w-4 h-4 text-emerald-600 flex-shrink-0" />}
                        {show && chosen && !isAns && <X className="w-4 h-4 text-red-500 flex-shrink-0" />}
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Fill in the gap */}
              {q.type === "fill" && (
                <div className="flex flex-col sm:flex-row gap-2">
                  <Input
                    disabled={show}
                    value={fillInputs[q.id] ?? ""}
                    onChange={(e) => setFillInputs((f) => ({ ...f, [q.id]: e.target.value }))}
                    onKeyDown={(e) => e.key === "Enter" && submitFill(q)}
                    placeholder="Type your answer…"
                    className="border-emerald-200 focus-visible:ring-emerald-500"
                  />
                  <Button
                    disabled={show}
                    onClick={() => submitFill(q)}
                    className="bg-emerald-700 hover:bg-emerald-800"
                  >
                    Submit
                  </Button>
                </div>
              )}

              {/* Hint */}
              {!show && (
                <button
                  onClick={() => setHintShown((h) => ({ ...h, [q.id]: true }))}
                  className="mt-3 text-xs text-emerald-700 hover:text-emerald-900 inline-flex items-center gap-1"
                >
                  <Lightbulb className="w-3.5 h-3.5" />
                  {hintShown[q.id] ? "Hint shown below" : "Need a hint?"}
                </button>
              )}
              {!show && hintShown[q.id] && (
                <p className="mt-2 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded px-3 py-2">
                  Topic: <strong>{TOPIC_LABELS[q.topic]}</strong> · Difficulty: <strong>{q.difficulty}</strong>.
                  Re-read the relevant module above before answering.
                </p>
              )}

              <AnimatePresence>
                {show && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className={`mt-4 text-sm rounded-lg px-4 py-3 border ${
                      correct
                        ? "bg-emerald-50 border-emerald-200 text-emerald-900"
                        : "bg-red-50/60 border-red-200 text-red-900"
                    }`}
                  >
                    <p className="font-semibold mb-1">
                      {correct ? "✓ Correct" : "✗ Not quite"}
                    </p>
                    <p>{q.explain}</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </Card>
          );
        })}
      </div>

      {/* Finish summary */}
      {finished && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden"
        >
          <Card className="p-6 sm:p-8 border-emerald-300 bg-gradient-to-br from-emerald-50 to-white text-center">
            <Trophy className="w-12 h-12 mx-auto text-amber-500 mb-3" />
            <h3 className="font-serif text-2xl font-bold mb-1">
              You scored {score} / {questions.length}
            </h3>
            <p className="text-muted-foreground mb-1">
              {Math.round((score / questions.length) * 100)}% mastery on this set
            </p>
            {timed && (
              <p className="text-xs text-muted-foreground mb-4">
                Completed in {mm}:{ss}
              </p>
            )}
            <Button onClick={regenerate} className="mt-3 bg-emerald-700 hover:bg-emerald-800">
              <RotateCcw className="w-4 h-4 mr-1.5" /> Retry with new questions
            </Button>
          </Card>
        </motion.div>
      )}
    </div>
  );
}

function isCorrect(q: Question, given: number | string | undefined) {
  if (given === undefined) return false;
  if (q.type === "fill") {
    const v = String(given).trim().toLowerCase();
    const main = String(q.answer).trim().toLowerCase();
    if (v === main) return true;
    return (q.acceptable ?? []).some((a) => a.toLowerCase() === v);
  }
  return given === q.answer;
}
