import re
import sys

file = sys.argv[1]

with open(file) as infile:
    text = infile.read()

def clean_corpus(text):
    text = re.sub(r".+[^.?!]\n", "", text)
    text = re.sub(r".+[^.?!]\u2028", "\n", text)
    text = re.sub(r".+[^.?!]\u2029", "\n", text)
    text = text.replace("\t", "")
    text = text.replace("\r", "")
    text = text.replace("  ", " ")
    text = text.replace("? ", "?")
    text = text.replace("! ", "!")
    text = text.replace('”', '')
    text = text.replace('"', '')
    text = text.replace('.)', ').')
    text = text.replace('?)', ')?')
    text = text.replace('!)', ')!')
    
    pattern = r"""
    .+?
    (
    [?!.]
    (?!\s*(?:et al|Dr|Mr|Ms|St)\b)
    (?=\s|$)
    (?<!\d)
    (?<!\b\w{1})
    )
    """
    #Match ., ?, !, that are not followed by a digit, or preceeded by Dr, St, etc
    split_text = [item.group().strip() for item in re.finditer(pattern, text, flags=re.VERBOSE)]
    output = "\n".join(split_text)
    output = re.sub(r"\n+", "\n", output)
    output = re.sub(r" +", " ", output)

    return output

with open("corpus.txt", "w") as outfile:
    outfile.write(clean_corpus(text))

