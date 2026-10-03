"""Render an evidence-based alert within Discord's embed limits."""

import discord


def _clip(text, limit):
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _plain(text):
    return discord.utils.escape_mentions(discord.utils.escape_markdown(str(text)))


def marker(decision):
    if decision.get("head_sha"):
        revision = "commit " + decision["head_sha"][:12]
    else:
        revision = "source revision " + decision["source_revision"][:12]
    return f"DupCheck decision {decision['decision_id']} | {revision}"


def build_alert(decision):
    if not decision["is_duplicate"] or decision["status"] != "completed":
        raise ValueError("Only completed duplicate decisions can become alerts")
    embed = discord.Embed(
        title=_clip(f"Potential duplication · {'PR #' if decision.get('pr_url') else 'PR ID '}{decision['pr_number']}: {_plain(decision['pr_title'])}", 256),
        url=decision["pr_url"],
        description=_clip(_plain(decision["reason"]), 1400) + "\n\nPlease look into the overlap and review whether the existing implementation can be reused or extended.",
        color=0xE9A23B,
    )
    embed.set_footer(text=marker(decision))
    author = decision.get("author_name") or decision["author_login"]
    embed.add_field(name="PR author", value=_clip(f"{_plain(author)} ({_plain(decision['author_login'])})", 500), inline=True)
    scopes = {"whole_project": "Whole project", "partial": "Part of a project", "unspecified": "Scope needs review"}
    embed.add_field(name="Overlap", value=scopes[decision["duplicate_kind"]], inline=True)
    embed.add_field(name="Topic", value=_clip(_plain(decision["topic"]), 500), inline=False)
    if decision.get("pr_description"):
        embed.add_field(name="PR description", value=_clip(_plain(decision["pr_description"]), 600), inline=False)
    if decision["repository"] != "team repository":
        embed.add_field(name="Repository", value=_clip(_plain(decision["repository"]), 256), inline=False)
    if decision.get("folder_names"):
        embed.add_field(name="PR folders", value=_clip(_plain(", ".join(decision["folder_names"])), 500), inline=False)
    for match in decision["matches"][:5]:
        parts = []
        if match.get("topic"):
            parts.append("Topic: " + _plain(match["topic"]))
        if match.get("evidence"):
            parts.append("Evidence: " + _plain(match["evidence"]))
        if match["pr_files"]:
            parts.append("PR files: " + _plain(", ".join(match["pr_files"][:5])))
        if match["existing_files"]:
            parts.append("Existing files: " + _plain(", ".join(match["existing_files"][:5])))
        if match.get("url"):
            parts.insert(0, "Existing implementation: " + match["url"])
        remaining = 5500 - len(embed) - min(len(match["project_name"]) + 7, 256)
        if remaining <= 0:
            break
        embed.add_field(name=_clip("Match: " + _plain(match["project_name"]), 256),
                        value=_clip("\n".join(parts) or "See the decision explanation above.", min(800, remaining)), inline=False)
    if len(decision["matches"]) > 5:
        embed.add_field(name="Additional matches", value=f"{len(decision['matches']) - 5} more matches are saved with this decision.", inline=False)
    embed.add_field(name="Review feedback", value="👍 The duplication flag is correct.\n👎 This is a false positive.\nChoose one; remove your reaction to withdraw it.", inline=False)
    user_id = decision.get("discord_user_id")
    content = f"<@{user_id}> please review this duplication flag." if user_id else None
    mentions = discord.AllowedMentions(everyone=False, roles=False, users=[discord.Object(id=int(user_id))] if user_id else False, replied_user=False)
    return {"content": content, "embed": embed, "allowed_mentions": mentions}


def preview_alert(decision):
    alert = build_alert(decision)
    return {"content": alert["content"], "embeds": [alert["embed"].to_dict()],
            "allowed_mentions": alert["allowed_mentions"].to_dict()}
