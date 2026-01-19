def evaluate_infraction(checker, sentence, list_rules):

    forbidden_words_used = checker.retrieve_forbidden_items(sentence, list_rules)
    is_forbidden = False if len(forbidden_words_used) == 0 else True

    return forbidden_words_used, is_forbidden


def binary_evaluation(original, prediction):
    if original == prediction:
        return False
    else:
        return True
