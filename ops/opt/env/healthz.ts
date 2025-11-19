import type { NextApiRequest, NextApiResponse } from "next";

export default function handler(_req: NextApiRequest, res: NextApiResponse) {
  res.status(200).json({
    ok: true,
    name: "SharpsSignal Web",
    version: process.env.NEXT_PUBLIC_APP_VERSION || "dev",
    ts: new Date().toISOString(),
  });
}
