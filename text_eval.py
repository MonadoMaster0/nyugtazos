import os

def readText():
    files = os.listdir('text')
    texts = []
    for file in files:
        with open(os.path.join('text',file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts


def eval(text, exclusionLimit = 5):
    """Returns a percentage value of utility charachters in text and a boolean value based on exclusion limit."""
    exclusion_chars = [':', '.', '%', ',', '/', '"', '\'', '-', ')', '(']
    wrong_level = 0
    wrong_levels = []
    for ex_ch in exclusion_chars:
        wrong_level += text.count(ex_ch)
    percentage = round(wrong_level/len(text)*100, 2)
    passed = True if percentage < exclusionLimit else False
    return percentage, passed

def main():
    text = readText()
    files = os.listdir('text')
    level = [eval(i) for i in text]
    for i, l in enumerate(level):
        print(f'{files[i]}\t{l[0]} %\t{l[1]}')

if __name__ == '__main__':
    main()