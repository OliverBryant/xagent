"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ChevronLeft, Loader2, Plus, Upload } from "lucide-react";

import { MarkdownEditor } from "@/components/skill-hub/markdown-editor";
import { useI18n } from "@/contexts/i18n-context";
import {
  UPLOAD_ERROR_MESSAGES,
  apiRequest,
  isJsonRecord,
  getApiErrorMessage,
  getUploadErrorMessage,
  parseApiResponse,
} from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";

/**
 * Create-new-skill page.
 *
 * Two inputs: ``name`` (used verbatim as the on-disk directory name
 * and as the skill's identifier in the SkillManager — the parser
 * ignores the frontmatter ``name`` field, so the dir name is the
 * source of truth) and the SKILL.md body.
 *
 * On success we redirect to ``/skill-hub/<name>`` so the user can
 * immediately see the parsed detail view.
 */

const STARTER_TEMPLATE = `---
description: One-line summary of what this skill does.
when_to_use: "Use this skill when the user wants to ..."
tags:
  - example
---

# My Skill

## Overview

Describe what this skill is for.

## When to Use

Spell out the scenarios where the agent should pick this skill.

## Execution Flow

1. First step
2. Second step
3. Final output
`;

export default function NewSkillPage() {
  const apiBase = getApiUrl();
  const router = useRouter();
  const { t } = useI18n();

  const [name, setName] = useState("");
  const [skillMd, setSkillMd] = useState(STARTER_TEMPLATE);
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // The backend regex matches client-side validation here so we can
  // disable the Create button before the user wastes a round trip.
  const nameValid = /^[A-Za-z0-9_-]+$/.test(name);
  const canSubmit = nameValid && skillMd.trim().length > 0 && !saving && !uploading;

  // Upload a .zip skill bundle or a bare SKILL.md. The backend resolves the
  // skill name (typed override → frontmatter name → zip root dir) and we
  // redirect to whatever name it reports back.
  const handleUpload = async (file: File) => {
    if (uploading || saving) return;
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      // Send the typed name as an override so the user, not the archive's
      // directory layout, has the final say over the skill name.
      if (nameValid) form.append("name", name);
      const res = await apiRequest(`${apiBase}/api/skill-hub/upload`, {
        method: "POST",
        body: form,
      });
      const parsed = await parseApiResponse(res);
      if (!res.ok) {
        setError(
          getUploadErrorMessage(res, parsed, {
            generic: t("skillHub.newSkill.uploadFailed", { status: String(res.status) }),
            tooLarge: t("skillHub.newSkill.uploadTooLarge"),
            proxy: t("skillHub.newSkill.uploadProxyError"),
          }),
        );
        return;
      }
      const uploaded = isJsonRecord(parsed.data) ? parsed.data : {};
      const uploadedName = typeof uploaded.name === "string" ? uploaded.name : "";
      router.push(uploadedName ? `/skill-hub/${encodeURIComponent(uploadedName)}` : "/skill-hub");
    } catch (e) {
      console.error(e);
      setError(t("skillHub.newSkill.uploadNetworkError"));
    } finally {
      setUploading(false);
    }
  };

  // Only one file becomes one skill, so say so rather than silently
  // dropping the rest of a multi-file drop.
  const handleFileList = (list: FileList | null) => {
    if (uploading || saving) return;
    const picked = list?.[0];
    // Clear any previous failure before reacting to a fresh selection, so a
    // stale message never sits above an unrelated new attempt.
    setError(null);
    if (!picked) return;
    if (list && list.length > 1) {
      setError(t("skillHub.newSkill.singleFileOnly", { name: picked.name }));
      return;
    }
    void handleUpload(picked);
  };

  const handleCreate = async () => {
    if (uploading || saving) return;
    setSaving(true);
    setError(null);
    try {
      const res = await apiRequest(`${apiBase}/api/skill-hub/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, skill_md: skillMd }),
      });
      if (!res.ok) {
        const parsed = await parseApiResponse(res);
        setError(
          getApiErrorMessage(
            res,
            parsed,
            t("skillHub.newSkill.createFailed", { status: String(res.status) }),
          ),
        );
        return;
      }
      // Skip back to detail page — the create response is summary-only
      // and the detail page will fetch the parsed content.
      router.push(`/skill-hub/${encodeURIComponent(name)}`);
    } catch (e) {
      console.error(e);
      setError(t("skillHub.newSkill.createNetworkError"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-background">
      <div className="mx-auto w-full flex-1 px-6 py-10">
        <Link
          href="/skill-hub"
          className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-4 w-4" /> {t("skillHub.newSkill.back")}
        </Link>

        <div className="mb-6 flex items-center justify-between gap-3">
          <h1 className="text-2xl font-bold tracking-tight">{t("skillHub.newSkill.title")}</h1>
          <button
            type="button"
            onClick={handleCreate}
            disabled={!canSubmit}
            className="inline-flex h-9 items-center gap-1.5 rounded-md bg-primary px-4 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
            {saving ? t("skillHub.newSkill.creating") : t("skillHub.newSkill.create")}
          </button>
        </div>

        <div
          role="button"
          tabIndex={0}
          aria-disabled={uploading || saving}
          onClick={() => !(uploading || saving) && fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key !== "Enter" && e.key !== " ") return;
            // Space would otherwise scroll the page.
            e.preventDefault();
            if (!(uploading || saving)) fileInputRef.current?.click();
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={(e) => {
            // Crossing into a child fires dragleave on the zone itself; without
            // this the highlight flickers for the whole drag.
            if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
            setDragOver(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFileList(e.dataTransfer.files);
          }}
          className={`mb-6 flex cursor-pointer items-center gap-3 rounded-lg border border-dashed p-4 transition-colors ${
            dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
          } ${uploading || saving ? "pointer-events-none opacity-60" : ""}`}
        >
          {uploading ? (
            <Loader2 className="h-5 w-5 shrink-0 animate-spin text-muted-foreground" />
          ) : (
            <Upload className="h-5 w-5 shrink-0 text-muted-foreground" />
          )}
          <div>
            <p className="text-sm font-medium">
              {uploading ? t("skillHub.newSkill.importing") : t("skillHub.newSkill.importTitle")}
            </p>
            <p className="text-xs text-muted-foreground">{t("skillHub.newSkill.importHint")}</p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip,.md"
            className="hidden"
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => {
              handleFileList(e.target.files);
              e.target.value = "";
            }}
          />
        </div>

        <div className="mb-4">
          <label className="mb-1 block text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            {t("skillHub.newSkill.skillName")}
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-skill"
            className="h-10 w-full rounded-md border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
          />
          <p className="mt-1 text-[11px] text-muted-foreground">
            {t("skillHub.newSkill.nameHint", { path: "~/.xagent/skills/" })}
          </p>
          {name && !nameValid && (
            <p className="mt-1 text-[11px] text-destructive">
              {t("skillHub.newSkill.nameInvalid", { pattern: "[A-Za-z0-9_-]+" })}
            </p>
          )}
        </div>

        <MarkdownEditor
          value={skillMd}
          onChange={setSkillMd}
          rows={26}
          placeholder={t("skillHub.newSkill.placeholder")}
        />

        {error && (
          <div className="mt-4 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
