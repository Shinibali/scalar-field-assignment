import re


# Estimate Size and Price from 'amount'
def parse_amount(amount):
    numbers = [int(x.replace(',', '')) for x in re.findall(r'\d{1,3}(?:,\d{3})*(?:-\d{1,3}(?:,\d{3})*)?', amount)]
    if len(numbers) == 1:
        return numbers[0]
    elif len(numbers) == 2:
        return (numbers[0] + numbers[1]) / 2  # taking avg
    return 0


def print_repeated_strings(dictionary, item):
    for key, val in dictionary.items():
        print(f"{item} for {key} trades: {val:.3f}")