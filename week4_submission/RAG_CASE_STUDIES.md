# RAG case studies

The excerpts below are taken from the existing DeepSeek evaluation response file. Retrieval is model-independent in this project, so the retrieved chunks are the same for Qwen, LLaMA, and DeepSeek. Text is shortened only for readability; the source and key fact are preserved.

## Case A — Relevant retrieval and correct answer

**Question (Q01):** “How long do we have to pay an invoice?”

**Retrieved context:** The top result is `payment_policy.txt` at similarity **0.7916**. It states: “Invoices must be paid within thirty days from the invoice date.” Other returned chunks include invoice-policy material about invoice fields and a payment-policy chunk about disputes.

**DeepSeek response:** “Thirty days from the invoice date.”

**Analysis:** The required fact is first in the context and the answer is concise and correct. This is the intended relationship: relevant retrieval → focused context → correct grounded response. The extra invoice chunks are harmless here, but they are not needed.

## Case B — Strong retrieval, but a hallucinated/contradictory answer

**Question (Q22):** “Can either party terminate the agreement immediately without a material breach?”

**Retrieved context:** `termination_policy.txt` is ranked first at **0.8303**, followed by another termination chunk and two contract-policy chunks. They state both that normal termination requires thirty days’ written notice and that immediate termination may occur only after a material breach remains uncorrected for fifteen days.

**Responses:**

- DeepSeek: “Yes, either party can terminate the agreement immediately without a material breach by providing thirty days written notice …”
- Qwen: the same unsupported “Yes” conclusion.
- LLaMA: the same unsupported “Yes” conclusion, then adds the material-breach condition.

**Analysis:** Retrieval and context quality are strong, but all three models merge two different conditions: ordinary termination with notice and immediate termination for an uncorrected breach. The answer reverses the meaning of “immediately.” This demonstrates that RAG does not guarantee factual correctness; generation and prompt design still matter.

## Case C — Important fact retrieved but omitted in generation

**Question (Q10):** “What obligations continue after termination?”

**Retrieved context:** The first four chunks are from `termination_policy.txt` and `contract_policy.txt`, at **0.7747**, **0.7678**, **0.7633**, and **0.7264**. The context explicitly includes confidentiality, intellectual property, and payments due before termination.

**DeepSeek response:** It names confidentiality and intellectual property but omits payments that became due before termination.

**Analysis:** The retrieval did not miss the information—the second and third chunks contain it. The answer is incomplete because generation selected only part of the available context. This separates context coverage from answer completeness.

## Case D — Partial retrieval noise and an underspecified request

**Question (Q24):** “What payment methods are accepted for invoices?”

**Retrieved context:** The first result is `payment_policy.txt` at **0.7293**, but it only says that customers should use a payment method “specified by the service provider.” Two of the four results are `invoice_policy.txt` chunks about invoice information, not accepted methods.

**DeepSeek response:** “Customers should make payments using one of the payment methods specified by the service provider.”

**Analysis:** The returned context is topically related but cannot answer which methods are accepted. The correct response should say that the knowledge base does not list them. This is an example of partial retrieval noise plus a model failing to express the knowledge-base limitation.

## Case E — Correct abstention from the provided context

**Question (Q23):** “What is the penalty percentage for late invoice payments?”

**Retrieved context:** Relevant `payment_policy.txt` chunks are returned at **0.7243**, **0.7041**, and **0.7024**. They say only that interest charges or late-payment charges may apply; no percentage appears.

**DeepSeek response:** “There is no mention of a penalty percentage in the provided context regarding late invoice payments.”

**Analysis:** This is the desired response to an unanswerable detail. Relevant retrieval → adequate context → explicit non-invention. It should be used as an example of grounded abstention.

## Overall RAG finding

The fixed retrieval layer successfully returns an expected policy source for every evaluated question (Recall@4 = 100%). Yet model answer accuracy ranges from 64% to 76%. Retrieval quality is necessary but insufficient: retrieval must expose the key fact, context must not overwhelm it with adjacent material, and the LLM must preserve conditions, negations, and uncertainty when generating the answer.
