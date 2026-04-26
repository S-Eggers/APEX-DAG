from ..core.types import PRType
from ..core.utils import add_id, remove_id


def remove_assignments(PRs_filtered: list[PRType]) -> list[PRType]:
    """
    Optimizes the WIR by removing raw assignment operations and collapsing the lineage edges.
    """
    assign_operations = [
        (pr_id, {"input": pr[0], "output": pr[3]})
        for pr_id, pr in enumerate(PRs_filtered)
        if pr[2] is not None and remove_id(pr[2]) == "Assign" and pr[1] is None
    ]
    added_prs = []
    indices_to_delete = []

    for assign_pr_id, assign in assign_operations:
        revelant_input_prs = []
        for pr_id, pr2 in enumerate(PRs_filtered):
            if pr2[3] == assign["input"]:
                revelant_input_prs.append(pr_id)
                indices_to_delete.append(pr_id)
            if pr2[0] == assign["input"]:
                for rel_input_id in revelant_input_prs:
                    added_prs.append(
                        (
                            PRs_filtered[rel_input_id][0],
                            PRs_filtered[rel_input_id][1],
                            PRs_filtered[rel_input_id][2],
                            pr2[3],
                        )
                    )
        indices_to_delete.append(assign_pr_id)

    PRs_filtered_new = [
        pr for i, pr in enumerate(PRs_filtered) if i not in indices_to_delete
    ]
    return PRs_filtered_new + added_prs


def filter_PRs(PRs: list[PRType]) -> list[PRType]:
    """
    Consolidates redundant operation nodes generated during tuple/list destructuring.
    """
    filtered_PRs = []
    problematic_operations = dict()
    operations = set([o for (_, _, o, _) in PRs])

    for I, c, p, O in PRs:
        if (
            c is not None
            and p in operations
            and O in operations
            and remove_id(p) == remove_id(O)
        ) and O not in problematic_operations:
            problematic_operations[O] = c

    for I, c, p, O in PRs:
        if p in problematic_operations and c is None:
            original_caller = problematic_operations[p]
            filtered_PRs.append((I, original_caller, p, O))
        elif O in problematic_operations and c is not None:
            continue
        else:
            filtered_PRs.append((I, c, p, O))

    return filtered_PRs


def fix_bibartie_issue_import_from(PRs: list[PRType]) -> list[PRType]:
    """
    Patches a specific Vamsa graph construction bug regarding 'ImportFrom' node connections.
    """
    filtered_PRs = []
    imported = {}
    for I, c, p, O in PRs:
        if remove_id(p) == "ImportFrom":
            imported[O] = O + add_id()
            filtered_PRs.append((I, c, p, imported[O]))
            filtered_PRs.append((None, imported[O], O, None))
        elif c in imported:
            filtered_PRs.append((I, imported[c], p, O))
        else:
            filtered_PRs.append((I, c, p, O))

    return filtered_PRs
