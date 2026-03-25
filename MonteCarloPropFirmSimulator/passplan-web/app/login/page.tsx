'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const router = useRouter();

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        try {
            const res = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            if (res.ok) {
                router.push('/passplan');
                return;
            }

            setError('Incorrect username or password');
        } catch {
            setError('Authentication failed');
        }
    };

    return (
        <div style={{
            width: '100%',
            height: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg, #0B0B0F 0%, #1a1a2e 100%)',
            fontFamily: "'Segoe UI', system-ui, sans-serif"
        }}>
            <div style={{
                background: '#0e0e12',
                border: '1px solid rgba(255, 122, 24, 0.3)',
                borderRadius: '12px',
                padding: '2rem',
                maxWidth: '400px',
                width: '90%',
                boxShadow: '0 0 40px rgba(255, 122, 24, 0.1)'
            }}>
                <h1 style={{
                    fontSize: '2rem',
                    fontWeight: 700,
                    color: '#fff',
                    marginBottom: '0.5rem',
                    textAlign: 'center'
                }}>
                    PassPlan
                </h1>
                <p style={{
                    fontSize: '0.9rem',
                    color: '#666',
                    textAlign: 'center',
                    marginBottom: '2rem'
                }}>
                    Enter credentials to access
                </p>

                <form onSubmit={handleLogin}>
                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{
                            display: 'block',
                            fontSize: '0.85rem',
                            color: '#999',
                            marginBottom: '0.5rem'
                        }}>
                            Username
                        </label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => {
                                setUsername(e.target.value);
                                setError('');
                            }}
                            placeholder="Enter username"
                            style={{
                                width: '100%',
                                padding: '0.75rem',
                                background: '#0B0B0F',
                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                borderRadius: '8px',
                                color: '#e0e0e0',
                                fontSize: '0.95rem',
                                boxSizing: 'border-box',
                                outline: 'none'
                            }}
                            onFocus={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(255, 122, 24, 0.5)';
                            }}
                            onBlur={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                            }}
                        />
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                        <label style={{
                            display: 'block',
                            fontSize: '0.85rem',
                            color: '#999',
                            marginBottom: '0.5rem'
                        }}>
                            Password
                        </label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => {
                                setPassword(e.target.value);
                                setError('');
                            }}
                            placeholder="Enter password"
                            style={{
                                width: '100%',
                                padding: '0.75rem',
                                background: '#0B0B0F',
                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                borderRadius: '8px',
                                color: '#e0e0e0',
                                fontSize: '0.95rem',
                                boxSizing: 'border-box',
                                outline: 'none'
                            }}
                            onFocus={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(255, 122, 24, 0.5)';
                            }}
                            onBlur={(e) => {
                                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                            }}
                        />
                    </div>

                    {error && (
                        <div style={{
                            background: '#3b0a0a',
                            border: '1px solid #7f1d1d',
                            borderRadius: '8px',
                            color: '#fca5a5',
                            fontSize: '0.85rem',
                            padding: '0.75rem',
                            marginBottom: '1rem',
                            textAlign: 'center'
                        }}>
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        style={{
                            width: '100%',
                            background: 'linear-gradient(135deg, #FF7A18 0%, #FFB347 100%)',
                            border: 'none',
                            borderRadius: '8px',
                            color: '#fff',
                            fontSize: '0.95rem',
                            fontWeight: 700,
                            padding: '0.75rem',
                            cursor: 'pointer',
                            transition: 'opacity 0.15s',
                            boxShadow: '0 0 20px rgba(255, 122, 24, 0.25)'
                        }}
                        onMouseOver={(e) => {
                            e.currentTarget.style.opacity = '0.9';
                        }}
                        onMouseOut={(e) => {
                            e.currentTarget.style.opacity = '1';
                        }}
                    >
                        Access PassPlan
                    </button>
                </form>

                <p style={{
                    fontSize: '0.75rem',
                    color: '#444',
                    textAlign: 'center',
                    marginTop: '1.5rem'
                }}>
                    Secure access - Runs locally on your machine
                </p>
            </div>
        </div>
    );
}
