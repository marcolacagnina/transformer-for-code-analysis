import ast
import astor
import builtins
import sys
import logging


class _CodeNormalizer(ast.NodeTransformer):


    def __init__(self):
        self.var_map = {}
        self.func_map = {}
        self.var_counter = 0
        self.func_counter = 0
        self.known_names = set(dir(builtins)) | set(sys.builtin_module_names) | {'exit', 'quit'}

    def visit_Import(self, node: ast.Import) -> ast.Import:
        for alias in node.names:
            self.known_names.add(alias.asname or alias.name)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        for alias in node.names:
            self.known_names.add(alias.asname or alias.name)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name not in self.known_names and node.name not in self.func_map:
            self.func_map[node.name] = f"func_{self.func_counter}"
            self.func_counter += 1
        node.name = self.func_map.get(node.name, node.name)


        for arg in node.args.args:
            if arg.arg not in self.known_names and arg.arg not in self.var_map:
                self.var_map[arg.arg] = f"var_{self.var_counter}"
                self.var_counter += 1
            arg.arg = self.var_map.get(arg.arg, arg.arg)

        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:

        if node.id in self.known_names:
            return node


        is_call = isinstance(getattr(node, 'ctx', None), ast.Load) and hasattr(self, 'parent') and isinstance(
            self.parent, ast.Call) and self.parent.func == node

        if is_call:
            if node.id not in self.func_map:
                self.func_map[node.id] = f"func_{self.func_counter}"
                self.func_counter += 1
            node.id = self.func_map[node.id]
        elif isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
            if node.id not in self.var_map:
                self.var_map[node.id] = f"var_{self.var_counter}"
                self.var_counter += 1
            node.id = self.var_map[node.id]

        return node

    def generic_visit(self, node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        item.parent = node
            elif isinstance(value, ast.AST):
                value.parent = node
        return super().generic_visit(node)


def normalize_code(code: str) -> str:

    try:
        tree = ast.parse(code)
        normalizer = _CodeNormalizer()
        normalized_tree = normalizer.visit(tree)
        ast.fix_missing_locations(normalized_tree)
        return astor.to_source(normalized_tree)
    except Exception as e:
        logging.warning(f"Failed Normalization: {e}")
        return code