'''
Auxiliary functions for SRL training.
'''

import numpy as np
import re


def init_transitions_simplified(tag_dict):
    '''
    This function initializes a tag transition table containing only
    the boundaries IOBES.
    '''
    tags = sorted(tag_dict, key=tag_dict.get)
    transitions = []

    def trans(tag, x):
        if tag in 'OES':
            return 0 if x in 'BOS' else -1000
        if tag in 'IB':
            return 0 if x in 'IE' else -1000
        raise ValueError('Unexpected tag: %s' % tag)

    for tag in tags:
        transitions.append([trans(tag, next_tag) for next_tag in tags])

    # initial transition
    transitions.append([
        0 if next_tag in 'BOS' else -1000 for next_tag in tags
    ])

    return np.array(transitions, np.float)


def init_transitions(tag_dict, scheme):
    '''
    This function initializes the tag transition table setting
    very low values for impossible transitions.

    :param tag_dict: The tag dictionary mapping tag names to the
        network output number.
    :param scheme: either iob or iobes.
    '''
    scheme = scheme.lower()
    assert scheme in ('iob', 'iobes'), 'Unknown tagging scheme: %s' % scheme
    transitions = []

    # since dict's are unordered, let's take the tags in the correct order
    tags = sorted(tag_dict, key=tag_dict.get)

    def trans(tag, x):
        if tag == 'O':
            # next tag can be O, V or any B
            return 0 if re.match('B|S|V', x) \
                else -1 if x == 'O' else -1000
        if tag[0] in 'IB':
            block = tag[2:]
            if scheme == 'iobes':
                # next tag can be I or E (same block)
                return block, 0 if re.match('(I|E)-%s' % block, x) else -1000

            return 0 if re.match('I-%s' % block, x) or \
                re.match('B-(?!%s)' % block, x) \
                else -1 if x == 'O' else -1000

        if tag[0] in 'ES':
            # next tag can be O, S (new block) or B (new block)
            block = tag[2:]
            return 0 if re.match('(S|B)-(?!%s)' % block, x) \
                else -1 if x == 'O' else -1000
        raise ValueError('Unknown tag: %s' % tag)

    # transitions between tags
    for tag in tags:
        transitions.append([trans(tag, next_tag) for next_tag in tags])

    # starting tag
    # it can be O or any B/S
    transitions.append([
        0 if next_tag[0] in 'OBS' else -1000 for next_tag in tags
    ])

    return np.array(transitions, np.float)
