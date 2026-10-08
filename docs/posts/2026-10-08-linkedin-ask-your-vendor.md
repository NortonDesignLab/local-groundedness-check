# LinkedIn post, 8 October 2026: ask your vendor

Another AI stack diagram went through my feed this week. It had thirty pretty logos, with four layers and lines between all of them. It looked like somebody ran a request through Gemini Notebook to create a diagram to look "AI Smart".

Tossing logos together with lines is exactly what I've been writing about as "ai slop" for the past year. As a practitioner who runs vLLM, Ray, Prometheus, a vector database, a router, and a chat front end, I recognize when a poster is putting up a wall of images and logos versus original thoughts. If you want to see what's missing, I'm posting my own Github repo to show how I arrived at my own conclusions.

When you meet with your vendor who is building your AI answer service, ask this question:

"How are you sure the answer isn't made up, even if it's coming from my own database?"

Listen for if they are using a second model that reads the finished answer back against your source documents before the answer reaches the screen. I'm agnostic about what to use here. Personally, I'm an open source fan and use LettuceDetect, which comes from KR Labs, a small Hungarian research lab. IBM's Granite Guardian is also published with an open weight version. Jev is another option making a lot of noise right now, if you're fine with your documents leaving your building. I run both LettuceDetect and Guardian on results before I see anything posted on my own internal chat engine. On 1,094 labelled items, those two plus a second vote from the 27B model I already run missed 75 fabrications. Guardian alone missed 155.

If the answer is "the model is very accurate" or "it only uses your data," that's the wrong answer. Push back and ask for a purpose built checking model. I have personally proven that retrieval with reranking applied is very good at finding the right documents. However, that alone does nothing to stop the model inventing a number on the way out.

To learn more, go to my Github: https://github.com/NortonDesignLab/local-groundedness-check

The code I publish publicly is under MIT licensing. All I ask is you give me credit on your reference list if you use it!

---

Numbers trace to [docs/BENCHMARK.md](../BENCHMARK.md), final configuration at n=1094 (v1.3.1, stock Qwen3.8-27B as the second vote).
