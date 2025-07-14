import argparse
from pprint import pprint
from tqdm import tqdm
import guidance
from guidance import models, gen, select

@guidance(stateless=True)
def adverb_diagnose(lm, adverb, sentence):
    return (
        lm
        + f"""
    You are an expert linguist.
    You are presented with a sentence that contains an adverb.
    Your task is to diagnose if the adverb is subject-oriented or manner.
    Follow these steps to complete the task:
        Step 1. Rewrite the sentence in a simplified form preserving the adverb and its modified.
        Step 2. Analyze the adverb's usage in the sentence.
            If you can paraphrase the sentence as "X did Y in a (adjective form of the adverb) manner."
        Step 3. Check if it sounds correctly.
            If it does, write 'manner', otherwise, write 'non-manner'.
    Examples:
    Adverb: Stupidly
    Sentence: He stupidly ran into the wall, and I couldn't stop him.
    <think>
    Here the adverb is stupidly, so I have to analyze its usage in the sentence.
    First I will rewrite the sentence as described in Step 1, I can say:
    Step 1. He stupidly ran into the wall.
    Now I will paraphrase the sentence as described in Step 2:
    Step 2. He ran into the wall in a stupid manner.
    Now I will check if it sounds correctly as described in Step 3:
    Step 3. Yes, it does sound correctly, so I will write 'manner'.
    </think>
    1) He stupidly ran into the wall.
    2) He ran into the wall in a stupid manner.
    Answer: manner
    ---
    Adverb: {adverb}
    Sentence: {sentence}
    <think>
    {gen(max_tokens=512, stop='</think>')}
    </think>
    1) {gen("s1", max_tokens=100)}
    """
    )


@guidance(stateless=True)
def rhetorical(lm, lemma, usage1, usage2):
    return (
        lm
        + f"""
    You are an expert linguist.
    You are presented with two sentences that both contain a shared word.
    Your task is to create and analyze zeugmas.
    Follow these steps to complete the task:
        Step 1. Rewrite the first sentence in a simplified form preserving the lemma.
        Step 2. Rewrite the second sentence in a simplified form preserving the lemma.
        Step 3. Write a sentence that joins both sentences using zeugma and the same shared word.
            If the construction doesn't make a bad pun, write same, otherwise, write different.
    Examples:
    Lemma: Plane
    Context 1: He loves planes and want to become a pilot.
    Context 2: The plane landed just now.
    <think>
    Here the lemma is plane, so I have to join both sentences using this word.
    First I will rewrite the sentences in a simplified form. For the first sentence, I can say:
    He loves planes.
    For the second sentence, I can say:
    The plane landed.
    Now I will join both sentences using zeugma:
    He loves planes, like the one that landed.
    The zeugma version doens't sound like a bad pun, so I will write 'same'.
    </think>
    1) He loves planes.
    2) The plane landed.
    3) He loves planes, like the one that landed.
    Conclusion: It doesn't make a bad pun.
    Answer: same
    ---
    Lemma: Cell
    Context 1: Anyone leaves a cell phone or handheld at home, many of them faculty members from nearby.
    Context 2: I just watch the dirty shadow the window bar makes across the wall of my cell.
    <think>
    Here the lemma is cell, so I have to join both sentences using this word.
    First I will rewrite the sentences in a simplified form. For the first sentence, I can say:
    Anyone leaves a cell phone at home.
    For the second sentence, I can say:
    The wall of my cell.
    Now I will join both sentences using zeugma:
    The wall of my cell which I leave at home.
    The zeugma version makes a bad pun, it doesn't sound right, so I will write 'different'.
    </think>
    1) Anyone leaves a cell phone at home.
    2) The wall of my cell.
    3) The wall of my cell which I leave at home.
    Conclusion: It makes a bad pun.
    Answer: different
    ---
    Lemma: {lemma}
    Context 1: {usage1}
    Context 2: {usage2}
    <think>
    {gen(stop='</think>')}
    </think>
    1) {gen("s1", max_tokens=100)}
    """
    )
# 2) {gen("s2", stop="\n")}
#     3) {gen("s3", stop="\n")}
#     Conclusion: {gen("conclude", stop="\n")}
#     Answer: {select(["same", "different"], "answer")}

def main(raw_args=None):
    parser = argparse.ArgumentParser(
        prog="prompt_generate.py",
        description="What the program does",
        epilog="""Generate the prompt for querying an LLM.

    Usage:
        prompt_generate.py [-n] <c1> <c2> <targets>

    Arguments:

        <corpus> = corpus
        <instruction> = give examples on how to do the task
        <task> = task to generate the prompt
        <model> = prompt the model to reason before answering

    """,
    )
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(raw_args)
    model = models.LlamaCpp(
        args.model,
        n_ctx=args.ctx,
        # n_gpu_layers=-1,
        # flash_attn=True,
        # echo=False,
    )

    lm = model
    # lemma = input("Lemma> ")
    # usage1 = input("Usage1> ")
    # usage2 = input("Usage2> ")
    # print(f"Lemma: {lemma}")
    # print(f"Usage1: {usage1}")
    # print(f"Usage2: {usage2}")
    # print("Generating prompt...")
    # lm += rhetorical(lemma, usage1, usage2)
    adverb = input("Adverb> ")
    sentence = input("Sentence> ")
    print(f"Adverb: {adverb}")
    print(f"Sentence: {sentence}")
    print("Generating prompt...")
    lm += adverb_diagnose(adverb, sentence)
    print(str(lm))
    # output = {
    #     "s1": lm["s1"],
    #     "s2": lm["s2"],
    #     "s3": lm["s3"],
    #     "conclude": lm["conclude"],
    #     "answer": lm["answer"],
    # }
    # pprint(output)


if __name__ == "__main__":
    main()
