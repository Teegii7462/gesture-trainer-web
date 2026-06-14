import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Gesture Trainer — beckon / shoo",
  description:
    "Record your hand gestures and train a shared beckon/shoo model in your browser.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
