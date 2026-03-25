import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
    const response = NextResponse.json({ success: true });
    response.cookies.set('simulator_auth', '', {
        httpOnly: true,
        maxAge: 0
    });
    return response;
}
