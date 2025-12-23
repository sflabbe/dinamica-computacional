# This is the optimized assemble() method - will be integrated into model.py

def assemble(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    nd = self.ndof()
    K = np.zeros((nd, nd))
    R = np.zeros(nd)
    info = {"hinges": []}

    # Collect trial axial forces (tension-positive convention) for each frame element.
    N_beam_trial: List[float] = []

    # Use JIT kernel if available (DC_FAST=1)
    use_jit = assemble_elements is not None

    if use_jit and len(self.beams) > 0:
        # Preallocate arrays for JIT kernel (beams)
        beam_k_locals = []
        beam_f_locals = []
        beam_dof_maps = []

        for e in self.beams:
            dofs, k_g, f_g, meta = e.stiffness_and_force_global(u_trial, include_geo=self.nlgeom)
            N_beam_trial.append(float(meta.get("N", 0.0)))
            beam_k_locals.append(k_g)
            beam_f_locals.append(f_g)
            beam_dof_maps.append(dofs)

        # Convert to numpy arrays and aggregate via JIT kernel
        beam_k_arr = np.array(beam_k_locals, dtype=float)
        beam_f_arr = np.array(beam_f_locals, dtype=float)
        beam_dof_arr = np.array(beam_dof_maps, dtype=np.int32)
        assemble_elements(K, R, beam_k_arr, beam_f_arr, beam_dof_arr)

    else:
        # Fallback: original Python loop
        for e in self.beams:
            dofs, k_g, f_g, meta = e.stiffness_and_force_global(u_trial, include_geo=self.nlgeom)
            N_beam_trial.append(float(meta.get("N", 0.0)))
            for a, ia in enumerate(dofs):
                R[ia] += f_g[a]
                for b, ib in enumerate(dofs):
                    K[ia, ib] += k_g[a, b]

    # [Axial coupling logic - unchanged, lines 72-103]
    for h in self.hinges:
        if isinstance(h, FiberRotSpringElement) and h.beam_idx is not None:
            bi = int(h.beam_idx)
            if 0 <= bi < len(N_beam_trial):
                N_tension = float(N_beam_trial[bi])
                h._beam_N_tension = N_tension
                h.hinge.N_target = float(h.beam_sign) * N_tension

        if isinstance(h, RotSpringElement) and h.beam_idx is not None:
            bi = int(h.beam_idx)
            if 0 <= bi < len(N_beam_trial):
                N_tension = float(N_beam_trial[bi])
                h._beam_N_tension = N_tension
                kind = str(getattr(h, "kind", "")).lower().strip()

                if kind == "beam_shm" and getattr(h, "beam_hinge", None) is not None:
                    N_comp = max(0.0, float(getattr(h, "beam_sign", -1.0)) * float(N_tension))
                    setattr(h.beam_hinge, "N_comp_current", float(N_comp))

                if kind in ("beam_fiber", "fiber"):
                    fh = getattr(h, "fiber_hinge", None) or getattr(h, "beam_hinge", None)
                    if fh is not None and hasattr(fh, "N_target"):
                        setattr(fh, "N_target", float(getattr(h, "beam_sign", -1.0)) * float(N_tension))

    # Hinges aggregation
    if use_jit and len(self.hinges) > 0:
        hinge_k_locals = []
        hinge_f_locals = []
        hinge_dof_maps = []

        for h in self.hinges:
            k_l, f_l, inf = h.eval_trial(u_trial, u_comm)
            dofs = h.dofs()
            hinge_k_locals.append(k_l)
            hinge_f_locals.append(f_l)
            hinge_dof_maps.append(np.array(dofs, dtype=np.int32))
            info["hinges"].append(inf)

        # Uniform size handling (pad if needed)
        max_dof = max(len(dofs) for dofs in hinge_dof_maps)
        hinge_k_padded = []
        hinge_f_padded = []
        hinge_dof_padded = []

        for k_l, f_l, dofs in zip(hinge_k_locals, hinge_f_locals, hinge_dof_maps):
            n_dof = len(dofs)
            if n_dof < max_dof:
                k_pad = np.zeros((max_dof, max_dof))
                k_pad[:n_dof, :n_dof] = k_l
                f_pad = np.zeros(max_dof)
                f_pad[:n_dof] = f_l
                dof_pad = np.zeros(max_dof, dtype=np.int32)
                dof_pad[:n_dof] = dofs
                hinge_k_padded.append(k_pad)
                hinge_f_padded.append(f_pad)
                hinge_dof_padded.append(dof_pad)
            else:
                hinge_k_padded.append(k_l)
                hinge_f_padded.append(f_l)
                hinge_dof_padded.append(dofs)

        hinge_k_arr = np.array(hinge_k_padded, dtype=float)
        hinge_f_arr = np.array(hinge_f_padded, dtype=float)
        hinge_dof_arr = np.array(hinge_dof_padded, dtype=np.int32)
        assemble_elements(K, R, hinge_k_arr, hinge_f_arr, hinge_dof_arr)

    else:
        # Fallback: original Python loop
        for h in self.hinges:
            k_l, f_l, inf = h.eval_trial(u_trial, u_comm)
            dofs = h.dofs()
            for a, ia in enumerate(dofs):
                R[ia] += f_l[a]
                for b, ib in enumerate(dofs):
                    K[ia, ib] += k_l[a, b]
            info["hinges"].append(inf)

    fd = self.free_dofs()
    return K[np.ix_(fd, fd)], R[fd], info
