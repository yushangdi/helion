"""Pallas-backend codegen for ops defined in ``helion.language.distributed_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
registrations; ``distributed_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.distributed_ops import start_async_remote_copy
from ...language.distributed_ops import wait_async_remote_copy
from ...language.distributed_ops import wait_send_async_remote_copy
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _codegen_at_expr(index_ast: object, prefix: str) -> tuple[str, dict[str, ast.AST]]:
    """Build a ``.at[...]`` subscript string + placeholder map from an index
    list.  Each element is either an ``int`` literal (emitted inline) or an
    already-lowered AST node (passed through as a placeholder).  ``prefix``
    namespaces the placeholders so src and dst indices never collide.
    """
    assert isinstance(index_ast, (list, tuple)), index_ast
    placeholders: dict[str, ast.AST] = {}
    parts: list[str] = []
    for i, elt in enumerate(index_ast):
        name = f"{prefix}{i}"
        if isinstance(elt, int):
            placeholders[name] = expr_from_string(repr(elt))
        else:
            assert isinstance(elt, ast.AST), elt
            placeholders[name] = elt
        parts.append(f"{{{name}}}")
    return ", ".join(parts), placeholders


@_decorators.codegen(start_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``_op = pltpu.make_async_remote_copy(...); _op.start()``.

    Emits two *independent* refs (``src.at[index]`` -> ``dst.at[dst_index]``)
    so the copy can be asymmetric.  ``device_id`` may be a runtime scalar.
    Stashes the ``_op`` variable name on the FX node's meta dict so the paired
    ``wait_async_remote_copy`` codegen can look it up.
    """
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    src_name = device_fn.tensor_arg(src).name
    dst_name = device_fn.tensor_arg(dst).name

    send_sem = device_fn.register_dma_semaphore(name_hint="send_sem")
    recv_sem = device_fn.register_dma_semaphore(name_hint="recv_sem")
    op_var = device_fn.new_var("_remote_copy", dce=False)

    # Stash the op variable name on the FX node so wait can find it.
    assert state.fx_node is not None
    state.fx_node.meta["_pallas_async_copy_op"] = op_var

    src_at, src_ph = _codegen_at_expr(state.ast_args[1], "_sidx")
    dst_at, dst_ph = _codegen_at_expr(state.ast_args[4], "_didx")

    device_id_ast = state.ast_args[2]
    if isinstance(device_id_ast, int):
        device_id_ast = expr_from_string(repr(device_id_ast))
    assert isinstance(device_id_ast, ast.AST)

    state.codegen.add_statement(
        statement_from_string(
            f"{op_var} = pltpu.make_async_remote_copy("
            f"{src_name}.at[{src_at}], {dst_name}.at[{dst_at}], "
            f"{send_sem}, {recv_sem}, "
            f"device_id={{device_id}}, "
            f"device_id_type=pl.DeviceIdType.LOGICAL)",
            device_id=device_id_ast,
            **src_ph,
            **dst_ph,
        )
    )
    state.codegen.add_statement(statement_from_string(f"{op_var}.start()"))
    return expr_from_string(op_var)


@_decorators.codegen(wait_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``<op_var>.wait()`` by looking up the op var stashed by
    the paired ``start_async_remote_copy`` on its FX node.
    """
    assert state.fx_node is not None
    descriptor_arg = state.fx_node.args[0]
    assert isinstance(descriptor_arg, torch.fx.Node), (
        "wait_async_remote_copy argument must be an FX node returned by "
        "start_async_remote_copy"
    )
    op_var = descriptor_arg.meta.get("_pallas_async_copy_op")
    if op_var is None:
        raise exc.InternalError(
            RuntimeError(
                "wait_async_remote_copy could not find the op variable name "
                "on the descriptor's source node — check that codegen for "
                "start_async_remote_copy ran first."
            )
        )
    state.codegen.add_statement(statement_from_string(f"{op_var}.wait()"))
    return expr_from_string("None")


@_decorators.codegen(wait_send_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``<op_var>.wait_send()`` (drain the local send) using the op var
    stashed by the paired ``start_async_remote_copy``."""
    assert state.fx_node is not None
    descriptor_arg = state.fx_node.args[0]
    assert isinstance(descriptor_arg, torch.fx.Node), (
        "wait_send_async_remote_copy argument must be an FX node returned by "
        "start_async_remote_copy"
    )
    op_var = descriptor_arg.meta.get("_pallas_async_copy_op")
    if op_var is None:
        raise exc.InternalError(
            RuntimeError(
                "wait_send_async_remote_copy could not find the op variable name "
                "on the descriptor's source node."
            )
        )
    state.codegen.add_statement(statement_from_string(f"{op_var}.wait_send()"))
    return expr_from_string("None")
