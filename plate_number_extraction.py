def find_letter_number_sequence(text):
    index = 0
    while index < len(text) - 1:
        # Check if the current character is a letter followed by a number
        if text[index].isalpha() and text[index + 1].isdigit():
            # Try to get a sequence of five characters starting from this point
            sequence = text[index:index + 6]
            # Check if we have exactly one letter followed by four numbers
            if len(sequence) == 6 and sequence[1:].isdigit():
                return sequence
        # Move to the next character
        index += 1
    return "No matching sequence found"

# Test example
text_string = " 	6B348824 h"
print(find_letter_number_sequence(text_string))  # Output should be "A41732"
