#!/usr/bin/env python3
from openai import AsyncOpenAI
import numpy as np
import asyncio
import sys

client = AsyncOpenAI()
concurrency = 20

async def main(input_file, output_file):
    """Takes a path to an input file (input_file), and
    an output file (output_file), and runs each line of text
    through ChatGPT to have it rewrite in its own style
    """
    with open(input_file) as infile:
        humanlines = infile.read().split("\n")
    
    output = []

    steps = range(len(humanlines))
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def limited_query(lines, id):
        async with semaphore:
                    print(f"Doing line {steps[id]}")
                    return await querygpt(humanlines[steps[id]])
            
    tasks = [limited_query(steps[i], i) for i in range(len(steps))]
    
    results = await asyncio.gather(*tasks)
    
    outputtext="\n".join(results)
    
    with open(output_file, "w") as outfile:
        outfile.write(outputtext)

    print("All done!")
    return 0


async def querygpt(text):
    """Takes a string that is a line of text, and returns
    a rewritten form created by ChatGPT
    """
    response = await client.responses.create(
        model="gpt-5-nano",
        input=f"The following text is a line of text you must rewrite in your own writing style as if it were in an essay; it cannot be the same as what was provided. Your only response should be the rewritten line, and nothing else.:\n\n {text}"
    )
    output = response.output_text
    
    while "\n\n" in output:
        output = output.replace("\n\n", "\n")
    return output



if len(sys.argv) != 3:
    print("Usage: rewrite.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

try:
    asyncio.run(main(input_file, output_file))
except KeyboardInterrupt:
    print("\nInterrupted by user.")
