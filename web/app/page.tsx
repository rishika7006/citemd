import type { AnswerRecord, EvalSummary, ServingSummary } from "@/lib/types";
import { Shell } from "@/components/shell";
import answersData from "@/public/demo/answers.json";
import evalData from "@/public/demo/eval.json";
import servingData from "@/public/demo/serving.json";

export default function Home() {
  return (
    <Shell
      answers={answersData as AnswerRecord[]}
      evalData={evalData as EvalSummary}
      servingData={servingData as ServingSummary}
    />
  );
}
