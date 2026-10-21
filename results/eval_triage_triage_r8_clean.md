| model | valid JSON | family acc | family F1 | severity acc | origin acc | all 3 right | noise filtered | incidents missed | median latency | prompt tokens | $ / 1k alerts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B, untuned, rules prompt | 9.0% | 6.0% | 0.077 | 3.0% | 9.0% | 1.0% | 0.0% | 9/81 | 2.14s | 886 | $0.0 |
| Llama 3.2 1B, untuned, rules + 3 examples | 98.0% | 56.0% | 0.471 | 31.0% | 94.0% | 22.0% | 42.1% | 8/81 | 2.58s | 1445 | $0.0 |
| **Llama 3.2 1B, fine-tuned (LoRA)** | 100.0% | 97.0% | 0.966 | 79.0% | 100.0% | 77.0% | 100.0% | 0/81 | 1.275s | 233 | $0.0 |
| Gemini 2.5 Flash, rules prompt | 100.0% | 100.0% | 1.0 | 54.0% | 100.0% | 54.0% | 100.0% | 0/81 | 1.126s | 925 | $0.4527 |
| Gemini 2.5 Flash + thinking, rules prompt | 99.0% | 99.0% | 0.996 | 69.0% | 99.0% | 69.0% | 100.0% | 0/81 | 4.815s | 925 | $2.1482 |
