import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
    const { username, password } = await req.json();

    if (username === 'ParamAkshar' && password === 'AlgoTrading123!') {
        const response = NextResponse.json({ success: true });
        response.cookies.set('simulator_auth', 'true', {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'lax',
            maxAge: 60 * 60 * 24 * 7
        });
        return response;
    }

    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
}
