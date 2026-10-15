import fs from 'fs';
import path from 'path';

// Async checkpoints (toolkit/async_checkpoint.py) publish a group of files and then mark
// <jobFolder>/.checkpoints/<group>.json as `completed`. Files listed by a manifest in any
// other state are half written or being deleted and must not be offered to the user.
// Files no manifest mentions are regular (synchronous / older) saves and stay visible.
export const getUncommittedCheckpointFiles = async (jobFolder: string): Promise<Set<string>> => {
  const uncommitted = new Set<string>();
  const manifestDir = path.join(jobFolder, '.checkpoints');
  let names: string[];
  try {
    names = await fs.promises.readdir(manifestDir);
  } catch {
    return uncommitted;
  }
  await Promise.all(
    names
      .filter(name => name.endsWith('.json'))
      .map(async name => {
        try {
          const manifest = JSON.parse(await fs.promises.readFile(path.join(manifestDir, name), 'utf-8'));
          if (manifest?.status === 'completed' || !Array.isArray(manifest?.files)) return;
          for (const file of manifest.files) {
            if (typeof file?.path === 'string') {
              uncommitted.add(path.resolve(jobFolder, file.path));
            }
          }
        } catch {
          // a manifest is replaced atomically, an unreadable one is not ours
        }
      }),
  );
  return uncommitted;
};
