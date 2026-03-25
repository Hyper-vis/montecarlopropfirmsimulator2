'use client';

import { useRouter } from 'next/navigation';

export function LogoutButton() {
    const router = useRouter();

    const handleLogout = async () => {
        await fetch('/api/logout', { method: 'POST' });
        router.push('/login');
    };

    return (
        <button
            onClick={handleLogout}
            style={{
                fontSize: '0.75rem',
                padding: '0.3rem 0.7rem',
                borderRadius: '6px',
                border: '1px solid rgba(255, 122, 24, 0.3)',
                background: 'transparent',
                color: '#FF9A45',
                cursor: 'pointer',
                transition: 'all 0.15s'
            }}
            onMouseOver={(e) => {
                e.currentTarget.style.background = 'rgba(255, 122, 24, 0.1)';
            }}
            onMouseOut={(e) => {
                e.currentTarget.style.background = 'transparent';
            }}
        >
            Logout
        </button>
    );
}
