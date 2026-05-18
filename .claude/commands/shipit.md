Commit all current changes and push to the remote repository. Follow these steps exactly:

1. Run `git status` to see what has changed (untracked files, modifications, deletions).
2. Run `git diff HEAD` to understand what specifically changed in the content.
3. Stage all changes with `git add -A`, but SKIP any files that look sensitive (e.g., .env, credentials, secrets).
4. Write a concise commit message (1 sentence) that describes what changed and why — focus on the "what changed" based on the actual diff, not a generic message.
5. Commit using a HEREDOC so formatting is preserved. Always append the co-author trailer:
   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
6. Push to origin with `git push`.
7. Report back: what files were committed, the commit message used, and confirm the push succeeded.

If there is nothing to commit (clean working tree), say so and do not create an empty commit.
