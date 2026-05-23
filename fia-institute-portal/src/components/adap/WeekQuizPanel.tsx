import { useState, useEffect } from "react";
import { ChevronDown, ChevronUp, CheckCircle, XCircle, RotateCcw, Trophy } from "lucide-react";
import { quizzes, type WeekKey } from "@/data/adapQuizData";

interface WeekQuizPanelProps {
  weekNumber: number;
  onPass: () => void;
  alreadyCompleted: boolean;
}

export function WeekQuizPanel({ weekNumber, onPass, alreadyCompleted }: WeekQuizPanelProps) {
  const weekKey = `week${weekNumber}` as WeekKey;
  const quiz = quizzes[weekKey];
  const [open, setOpen] = useState(false);
  const [currentQ, setCurrentQ] = useState(0);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [submitted, setSubmitted] = useState(false);
  const [score, setScore] = useState<number | null>(null);
  const [showReview, setShowReview] = useState(false);

  // Load previous best from localStorage
  const storageKey = `fia_quiz_${weekKey}`;
  const [highScore, setHighScore] = useState<number | null>(null);

  useEffect(() => {
    const raw = localStorage.getItem(storageKey);
    if (raw) {
      const d = JSON.parse(raw);
      setHighScore(d.highScore);
    }
  }, [storageKey]);

  const isPassed = (highScore ?? 0) >= quiz.passingScore || alreadyCompleted;
  const question = quiz.questions[currentQ];
  const allAnswered = quiz.questions.every((_, i) => answers[i] !== undefined);

  const handleSubmit = () => {
    let correct = 0;
    quiz.questions.forEach((q, i) => {
      if (answers[i] === q.correct) correct++;
    });
    setScore(correct);
    setSubmitted(true);

    const raw = localStorage.getItem(storageKey);
    const prev = raw ? JSON.parse(raw) : { highScore: 0, attempts: 0 };
    const newHigh = Math.max(correct, prev.highScore || 0);
    const newAttempts = (prev.attempts || 0) + 1;
    const passed = newHigh >= quiz.passingScore;

    localStorage.setItem(storageKey, JSON.stringify({ highScore: newHigh, attempts: newAttempts, passed }));
    setHighScore(newHigh);

    if (correct >= quiz.passingScore && !alreadyCompleted) {
      onPass();
    }
  };

  const handleRetake = () => {
    setAnswers({});
    setSubmitted(false);
    setScore(null);
    setShowReview(false);
    setCurrentQ(0);
  };

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: "1px solid #DDE8E3" }}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium"
        style={{ background: isPassed ? "#d4edda" : "#EAF7EF", color: "#0D5C3A" }}
      >
        <span className="flex items-center gap-2">
          📝 Week {weekNumber} Quiz
          {isPassed && <CheckCircle className="w-4 h-4" style={{ color: "#1B7A4E" }} />}
          {highScore !== null && (
            <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "rgba(13,92,58,0.1)", color: "#0D5C3A" }}>
              Best: {highScore}/10
            </span>
          )}
        </span>
        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {open && (
        <div className="bg-white p-4 space-y-4">
          <div className="text-center mb-2">
            <h3 className="font-serif font-bold text-base" style={{ color: "#0D5C3A" }}>{quiz.title}</h3>
            <p className="text-xs" style={{ color: "#4A6B5D" }}>{quiz.description} · Pass: ≥{quiz.passingScore}/10</p>
          </div>

          {/* Results */}
          {submitted && score !== null && (
            <div
              className="rounded-xl p-5 text-center space-y-3"
              style={{
                background: score >= quiz.passingScore ? "#EAF7EF" : "#FEF2F2",
                border: `2px solid ${score >= quiz.passingScore ? "#1B7A4E" : "#DC2626"}`,
              }}
            >
              {score >= quiz.passingScore ? (
                <>
                  <Trophy className="w-10 h-10 mx-auto" style={{ color: "#E8A020" }} />
                  <h3 className="text-lg font-bold" style={{ color: "#0D5C3A" }}>
                    🎉 You scored {score}/10 — Passed!
                  </h3>
                  <p className="text-sm" style={{ color: "#4A6B5D" }}>
                    {weekNumber < 6 ? `Week ${weekNumber + 1} is now unlocked!` : "All weeks complete! 🎓"}
                  </p>
                </>
              ) : (
                <>
                  <XCircle className="w-10 h-10 mx-auto text-red-500" />
                  <h3 className="text-lg font-bold" style={{ color: "#DC2626" }}>
                    You scored {score}/10
                  </h3>
                  <p className="text-sm" style={{ color: "#4A6B5D" }}>
                    Need ≥{quiz.passingScore}/10 to pass. Review and try again!
                  </p>
                </>
              )}
              <div className="flex gap-2 justify-center pt-1">
                <button
                  onClick={() => setShowReview(!showReview)}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors"
                  style={{ borderColor: "#DDE8E3", color: "#0D5C3A" }}
                >
                  {showReview ? "Hide" : "Review"} Answers
                </button>
                <button
                  onClick={handleRetake}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium text-white flex items-center gap-1"
                  style={{ background: "#1B7A4E" }}
                >
                  <RotateCcw className="w-3 h-3" /> Retake
                </button>
              </div>
            </div>
          )}

          {/* Review */}
          {showReview && submitted && (
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {quiz.questions.map((q, idx) => {
                const userAns = answers[idx];
                const isCorrect = userAns === q.correct;
                return (
                  <div
                    key={q.id}
                    className="rounded-lg p-3 text-sm space-y-1"
                    style={{
                      background: isCorrect ? "#f0fdf4" : "#fef2f2",
                      borderLeft: `3px solid ${isCorrect ? "#1B7A4E" : "#DC2626"}`,
                    }}
                  >
                    <div className="flex items-start gap-1.5">
                      {isCorrect ? (
                        <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "#1B7A4E" }} />
                      ) : (
                        <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5 text-red-500" />
                      )}
                      <span className="font-medium" style={{ color: "#0D5C3A" }}>{idx + 1}. {q.question}</span>
                    </div>
                    {!isCorrect && (
                      <p className="ml-5.5 text-xs">
                        <span style={{ color: "#DC2626" }}>Your answer: {q.options[userAns]}</span>
                        <br />
                        <span style={{ color: "#1B7A4E" }}>Correct: {q.options[q.correct]}</span>
                      </p>
                    )}
                    <p className="ml-5.5 text-xs italic" style={{ color: "#4A6B5D" }}>{q.explanation}</p>
                  </div>
                );
              })}
            </div>
          )}

          {/* Question */}
          {!submitted && (
            <div className="space-y-4">
              {/* Progress dots */}
              <div className="flex gap-1 justify-center flex-wrap">
                {quiz.questions.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setCurrentQ(i)}
                    className="w-7 h-7 rounded-full text-xs font-medium transition-colors"
                    style={{
                      background: i === currentQ ? "#0D5C3A" : answers[i] !== undefined ? "#1B7A4E" : "#EAF7EF",
                      color: i === currentQ || answers[i] !== undefined ? "white" : "#0D5C3A",
                    }}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>

              {/* Question text */}
              <div>
                <p className="font-medium text-sm mb-3" style={{ color: "#0D5C3A" }}>
                  <span style={{ color: "#4A6B5D" }}>Q{currentQ + 1}.</span> {question.question}
                </p>
                <div className="space-y-2">
                  {question.options.map((opt, i) => (
                    <button
                      key={i}
                      onClick={() => setAnswers(prev => ({ ...prev, [currentQ]: i }))}
                      className="w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all"
                      style={{
                        background: answers[currentQ] === i ? "#1B7A4E" : "#F5F8F6",
                        color: answers[currentQ] === i ? "white" : "#0D5C3A",
                        border: `1px solid ${answers[currentQ] === i ? "#1B7A4E" : "#DDE8E3"}`,
                      }}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              </div>

              {/* Nav */}
              <div className="flex items-center justify-between pt-2" style={{ borderTop: "1px solid #DDE8E3" }}>
                <button
                  onClick={() => setCurrentQ(p => Math.max(0, p - 1))}
                  disabled={currentQ === 0}
                  className="text-xs font-medium px-3 py-1.5 rounded-lg disabled:opacity-30"
                  style={{ color: "#0D5C3A" }}
                >
                  ← Previous
                </button>
                {currentQ < quiz.questions.length - 1 ? (
                  <button
                    onClick={() => setCurrentQ(p => p + 1)}
                    className="text-xs font-medium px-3 py-1.5 rounded-lg"
                    style={{ color: "#0D5C3A" }}
                  >
                    Next →
                  </button>
                ) : (
                  <button
                    onClick={handleSubmit}
                    disabled={!allAnswered}
                    className="text-xs font-medium px-4 py-1.5 rounded-lg text-white disabled:opacity-40"
                    style={{ background: "#0D5C3A" }}
                  >
                    Submit Quiz
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
