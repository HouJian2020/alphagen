from graphviz import Digraph


def split_string_with_ignore_parentheses(formula_str):
    result = []
    current = ""
    parentheses_count = 0

    for char in formula_str:
        if char == "(":
            parentheses_count += 1
        elif char == ")":
            parentheses_count = max(0, parentheses_count - 1)
        elif char == "," and parentheses_count == 0:
            result.append(current)
            current = ""
            continue
        current += char

    if current:
        result.append(current)

    return result


def parse_formula(formula, show_nodeid=False):
    # 定义节点ID计数器
    node_id_counter = 0

    def parse_helper(formula_str, dot, node_id):
        nonlocal node_id_counter
        node_name = f"node_id {node_id}"
        label_prefix = node_name + "\n" if show_nodeid else ""
        # 按照公式中的逗号和括号拆分公式
        if ("(" in formula_str) and (not formula_str.startswith("Constant")):
            if ")" != formula_str[-1]:
                raise ValueError("公式不合法: " + formula_str)

            token_name, sub_tokens = formula_str.split("(", maxsplit=1)
            sub_tokens = split_string_with_ignore_parentheses(sub_tokens[:-1])

            dot.node(
                name=node_name,
                label=label_prefix + token_name,
                fillcolor="#136ed4",
                style="filled",
            )

            for sub_formular in sub_tokens:
                node_id_counter += 1
                sub_node_name = f"node_id {node_id_counter}"
                dot.node(name=sub_node_name)
                dot.edge(node_name, sub_node_name)
                parse_helper(sub_formular, dot, node_id_counter)

        else:
            if formula_str.startswith("Constant"):
                formula_str = formula_str.split('(')[1][:-1]
            if formula_str.startswith("$"):
                formula_str = formula_str[1:]
            dot.node(
                name=node_name,
                label=label_prefix + formula_str,
                fillcolor="lightblue",
                style="filled",
            )

    # 创建图
    dot = Digraph(comment="Formula Graph")

    # 解析并构建图
    parse_helper(formula, dot, 0)

    return dot