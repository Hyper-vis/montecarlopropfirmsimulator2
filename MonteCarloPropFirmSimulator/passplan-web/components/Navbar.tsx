"use client";

import { useState } from "react";
import Link from "next/link";
import { Menu, X, BarChart2 } from "lucide-react";

export default function Navbar() {
    const [open, setOpen] = useState(false);

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-[#0B0B0F]/80 backdrop-blur-xl">
            <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                {/* Logo */}
                <a href="/" className="flex items-center gap-2.5 group">
                    <div className="w-8 h-8 rounded-lg bg-accent-gradient flex items-center justify-center shadow-accent-glow-sm">
                        <BarChart2 className="w-4 h-4 text-white" strokeWidth={2.5} />
                    </div>
                    <span className="font-bold text-white text-lg tracking-tight">
                        Pass<span className="text-gradient">Plan</span>
                    </span>
                </a>

                {/* Desktop nav */}
                <div className="hidden md:flex items-center gap-8">
                    <a href="#product" className="text-sm text-white/50 hover:text-white transition-colors duration-200">
                        Product
                    </a>
                    <a href="#how-it-works" className="text-sm text-white/50 hover:text-white transition-colors duration-200">
                        How it works
                    </a>
                    <a href="#docs" className="text-sm text-white/50 hover:text-white transition-colors duration-200">
                        Docs
                    </a>
                    <a href="/discover" className="text-sm text-white/50 hover:text-white transition-colors duration-200">
                        Discover
                    </a>
                    <Link
                        href="/passplan"
                        className="btn-glow inline-flex items-center gap-2 rounded-lg bg-accent-gradient px-4 py-2 text-sm font-semibold text-white transition-all duration-300"
                    >
                        Open PassPlan
                    </Link>
                </div>

                {/* Mobile menu button */}
                <button
                    className="md:hidden text-white/60 hover:text-white transition-colors"
                    onClick={() => setOpen(!open)}
                >
                    {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                </button>
            </div>

            {/* Mobile menu */}
            {open && (
                <div className="md:hidden border-t border-white/5 bg-[#0B0B0F]">
                    <div className="max-w-7xl mx-auto px-6 py-4 flex flex-col gap-4">
                        <a href="#product" className="text-sm text-white/60 hover:text-white transition-colors">Product</a>
                        <a href="#how-it-works" className="text-sm text-white/60 hover:text-white transition-colors">How it works</a>
                        <a href="#docs" className="text-sm text-white/60 hover:text-white transition-colors">Docs</a>
                        <a href="/discover" className="text-sm text-white/60 hover:text-white transition-colors">Discover</a>
                        <Link
                            href="/passplan"
                            className="btn-glow inline-flex justify-center rounded-lg bg-accent-gradient px-4 py-2.5 text-sm font-semibold text-white"
                        >
                            Open PassPlan
                        </Link>
                    </div>
                </div>
            )}
        </nav>
    );
}
