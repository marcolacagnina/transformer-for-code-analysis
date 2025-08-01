import logging
import ast
import astor


class _CodeBlockTagger(ast.NodeVisitor):

    def __init__(self):
        self.blocks = {"imports": [], "functions": [], "global": []}

    def visit(self, node):
        global_block = []
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                try:
                    self.blocks["imports"].append(astor.to_source(stmt).strip())
                except Exception:
                    continue
            elif isinstance(stmt, ast.FunctionDef):
                self.blocks["functions"].append(astor.to_source(stmt).strip())
            else:
                try:
                    global_block.append(astor.to_source(stmt).strip())
                except Exception:
                    continue

        if global_block:
            self.blocks["global"].append("\n".join(global_block))


def tag_code_blocks(code: str) -> str:
    try:
        tree = ast.parse(code)
        tagger = _CodeBlockTagger()
        tagger.visit(tree)

        parts = []
        if tagger.blocks["imports"]:
            parts.append("<IMPORTS_START>\n" + "\n".join(tagger.blocks["imports"]) + "\n<IMPORTS_END>")

        for func in tagger.blocks["functions"]:
            parts.append(f"<FUNC_DEF_START>\n{func}\n<FUNC_DEF_END>")

        for glob in tagger.blocks["global"]:
            parts.append(f"<GLOBAL_CODE_START>\n{glob}\n<GLOBAL_CODE_END>")

        return "\n\n".join(parts)
    except Exception as e:
        logging.warning(f"Failed Parsing: {e}")
        return code