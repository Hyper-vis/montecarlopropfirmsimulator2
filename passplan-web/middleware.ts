import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
    // If accessing simulator routes, check auth
    if (request.nextUrl.pathname.startsWith('/simulator')) {
        const auth = request.cookies.get('simulator_auth')?.value;

        if (!auth) {
            // Redirect to login
            return NextResponse.redirect(new URL('/login', request.url));
        }
    }

    return NextResponse.next();
}

export const config = {
    matcher: ['/simulator/:path*'],
};
