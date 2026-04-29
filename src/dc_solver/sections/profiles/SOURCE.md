# Steel profile mini-database source note

This folder contains a **small helper dataset** for IPE/HEA/HEB sections (100/200/300 only).

- Intended use: section property helper and geometric precheck workflows.
- Not intended use: normative design database.
- Each JSON record keeps original units (`mm, cm2, cm4, cm3`) and a `review_required: true` flag.
- Data origin: consolidated from public manufacturer profile tables; users must verify against approved project references before production use.

Unit conversion on load:
- `mm -> m`: `1e-3`
- `cm2 -> m2`: `1e-4`
- `cm3 -> m3`: `1e-6`
- `cm4 -> m4`: `1e-8`
