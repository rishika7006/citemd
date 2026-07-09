"use client";

import { useState } from "react";
import { LINKS } from "@/lib/links";
import { GithubIcon, LinkedinIcon, MailIcon } from "@/components/icons";

export function RepoLink() {
  return (
    <a
      href={LINKS.repo}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink transition-colors"
    >
      <GithubIcon className="w-3.5 h-3.5" />
      Repository
    </a>
  );
}

function MenuItem({
  href,
  icon,
  label,
}: {
  href: string;
  icon: React.ReactNode;
  label: string;
}) {
  const external = href.startsWith("http");
  return (
    <a
      href={href}
      {...(external ? { target: "_blank", rel: "noreferrer" } : {})}
      className="flex items-center gap-2.5 px-3 py-2 text-sm text-muted hover:text-ink hover:bg-sunk transition-colors"
    >
      <span className="text-faint">{icon}</span>
      {label}
    </a>
  );
}

export function ContactMenu() {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="menu"
        className="inline-flex items-center gap-1 text-sm text-muted hover:text-ink transition-colors"
      >
        Contact
        <span className="text-[9px] leading-none mt-0.5">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <>
          <button
            aria-hidden="true"
            tabIndex={-1}
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-10 cursor-default"
          />
          <div
            role="menu"
            className="absolute right-0 mt-2 z-20 w-52 rounded-md border border-line bg-raised py-1 shadow-lg"
          >
            <MenuItem href={`mailto:${LINKS.email}`} icon={<MailIcon />} label="Email" />
            <MenuItem href={LINKS.linkedin} icon={<LinkedinIcon />} label="LinkedIn" />
            <MenuItem href={LINKS.profile} icon={<GithubIcon />} label="GitHub profile" />
          </div>
        </>
      )}
    </div>
  );
}
