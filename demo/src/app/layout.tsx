import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Indic Transliteration Demo",
  description: "Live Latin-to-Devanagari suggestions for courtroom typing",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
