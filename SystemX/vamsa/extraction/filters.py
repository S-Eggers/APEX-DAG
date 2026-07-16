from ..core.types import PRType
from ..core.utils import add_id, remove_id

def remove_assignments(prs_filtered: list[PRType]) -> list[PRType]:
    """Optimizes the WIR by removing raw assignment operations and collapsing the lineage edges."""
    assign_operations = [(pr_id, {"input": pr[0], "output": pr[3]}) for pr_id, pr in enumerate(prs_filtered) if pr[2] is not None and remove_id(pr[2]) == "Assign" and pr[1] is None]
    added_prs = []
    indices_to_delete = []

    for assign_pr_id, assign in assign_operations:
        revelant_input_prs = []
        for pr_id, pr2 in enumerate(prs_filtered):
            if pr2[3] == assign["input"]:
                revelant_input_prs.append(pr_id)
                indices_to_delete.append(pr_id)
            if pr2[0] == assign["input"]:
                for rel_input_id in revelant_input_prs:
                    added_prs.append(
                        (
                            prs_filtered[rel_input_id][0],
                            prs_filtered[rel_input_id][1],
                            prs_filtered[rel_input_id][2],
                            pr2[3],
                        )
                    )
        indices_to_delete.append(assign_pr_id)

    prs_filtered_new = [pr for i, pr in enumerate(prs_filtered) if i not in indices_to_delete]
    return prs_filtered_new + added_prs

def filter_prs(prs: list[PRType]) -> list[PRType]:
    """Consolidates redundant operation nodes generated during tuple/list destructuring."""
    filtered_prs = []
    problematic_operations: dict = {}
    operations = {o for (_, _, o, _) in prs}

    for _inp, c, p, out in prs:
        if (c is not None and p in operations and out in operations and remove_id(p) == remove_id(out)) and out not in problematic_operations:
            problematic_operations[out] = c

    for inp, c, p, out in prs:
        if p in problematic_operations and c is None:
            original_caller = problematic_operations[p]
            filtered_prs.append((inp, original_caller, p, out))
        elif out in problematic_operations and c is not None:
            continue
        else:
            filtered_prs.append((inp, c, p, out))

    return filtered_prs

filter_PRs = filter_prs  # noqa: N816

def fix_bibartie_issue_import_from(prs: list[PRType]) -> list[PRType]:
    """Patches a specific Vamsa graph construction bug regarding 'ImportFrom' node connections."""
    filtered_prs = []
    imported: dict = {}
    for inp, c, p, out in prs:
        if remove_id(p) == "ImportFrom":
            imported[out] = out + add_id()
            filtered_prs.append((inp, c, p, imported[out]))
            filtered_prs.append((None, imported[out], out, None))
        elif c in imported:
            filtered_prs.append((inp, imported[c], p, out))
        else:
            filtered_prs.append((inp, c, p, out))

    return filtered_prs
