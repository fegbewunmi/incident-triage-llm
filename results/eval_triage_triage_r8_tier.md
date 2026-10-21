| model | valid JSON | family acc | family F1 | severity acc | origin acc | all 3 right | noise filtered | incidents missed | median latency | prompt tokens | $ / 1k alerts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B, untuned, rules prompt | 8.0% | 4.0% | 0.054 | 3.0% | 7.0% | 1.0% | 0.0% | 8/81 | 2.214s | 957 | $0.0 |
| Llama 3.2 1B, untuned, rules + 3 examples | 93.0% | 53.0% | 0.451 | 38.0% | 87.0% | 26.0% | 52.6% | 15/81 | 2.621s | 1534 | $0.0 |
| **Llama 3.2 1B, fine-tuned (LoRA)** | 100.0% | 96.0% | 0.955 | 87.0% | 100.0% | 84.0% | 100.0% | 0/81 | 1.301s | 240 | $0.0 |
| Gemini 2.5 Flash, rules prompt | 100.0% | 98.0% | 0.982 | 70.0% | 99.0% | 68.0% | 100.0% | 2/81 | 1.248s | 1002 | $0.4789 |
| Gemini 2.5 Flash + thinking, rules prompt | 99.0% | 99.0% | 0.995 | 96.0% | 99.0% | 96.0% | 100.0% | 0/81 | 5.426s | 1002 | $2.2366 |
