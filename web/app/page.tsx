import type { AnswerRecord, EvalSummary } from "@/lib/types";
import { Shell } from "@/components/shell";
import answersData from "@/public/demo/answers.json";
import evalData from "@/public/demo/eval.json";

export default function Home() {
  return <Shell answers={answersData as AnswerRecord[]} evalData={evalData as EvalSummary} />;
}
