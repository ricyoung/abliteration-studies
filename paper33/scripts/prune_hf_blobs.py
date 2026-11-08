"""Delete files in the SHARED hub/blobs dir that no snapshot link (of any model/dataset) resolves to.
Per-repo blobs dirs are never touched. Dry run unless --delete."""
import os, sys, glob
hub = os.environ.get("HF_HOME", "/workspace/paper33/hf") + "/hub"
shared = os.path.join(hub, "blobs") + os.sep
keep = {os.path.realpath(l) for l in glob.glob(hub + "/*/snapshots/*/**", recursive=True) if os.path.islink(l)}
kept_shared = {k for k in keep if k.startswith(shared)}
orph = [b for b in glob.glob(shared + "**", recursive=True)
        if os.path.isfile(b) and not os.path.islink(b) and not b.endswith(".lock") and os.path.realpath(b) not in keep]
print(f"links resolved: {len(keep)} (of which in shared blobs: {len(kept_shared)}, {sum(os.path.getsize(k) for k in kept_shared)/1e9:.1f} GB kept)")
print(f"orphans in shared blobs: {len(orph)} files, {sum(os.path.getsize(b) for b in orph)/1e9:.1f} GB")
print("orphan/kept overlap:", len({os.path.realpath(b) for b in orph} & keep), "| all orphans inside shared dir:", all(b.startswith(shared) for b in orph))
if "--delete" in sys.argv:
    for b in orph: os.remove(b)
    print("deleted", len(orph))
