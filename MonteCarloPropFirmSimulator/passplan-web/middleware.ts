import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
    // If accessing protected product routes, check auth
    if (request.nextUrl.pathname.startsWith('/simulator') || request.nextUrl.pathname.startsWith('/passplan')) {
        const auth = request.cookies.get('simulator_auth')?.value;

        if (!auth) {
            // Redirect to login
            return NextResponse.redirect(new URL('/login', request.url));
        }
    }

    return NextResponse.next();
}

export const config = {
    matcher: ['/simulator/:path*', '/passplan/:path*'],
};
