import re
import pprint
from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
import pandas as pd

tester = """
Annual Report to Congress:  Military  and Security Developments Involving the People’s Republic of
China. This is what a sentence looks like. This is more than one sentence. this is THIS 2222 HH iS
NONESENCE."""

grammar = Grammar(
    r"""
    sentence = (word punct ws?)+
    word = ~r"[a-zA-Z]+"
    punct = ~r"[.?!]"
    ws = ~r"\s*"
    """
    """re write as a single expression from above
    total = ([a-zA-Z]+[.?!]\s*?)+
    (word punct ws?)+[]

    """
)
class sentenceParserVisitor(NodeVisitor):
    def visit_expr(self, node, visited_children):
        """return the result of the first visited child"""
        return visited_children[0]

    def visit_sentence(self, node, visited_children):
        sentence = ""
        for child in visited_children:
            if isinstance(child, str):
                sentence += child
        return sentence.strip()

    def visit_word(self, node, visited_children):
        return node.text

    def visit_punct(self, node, visited_children):
        return node.text

    def visit_ws(self, node, visited_children):
        return node.text

    def generic_visit(self, node, visited_children):
        return visited_children or node


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    parse_tree = grammar["sentence"].parse(tester)
    iv = SentenceParserVisitor()
    sentences = iv.visit(parse_tree)
    print(sentences)



    
