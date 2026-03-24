'use client';

import { LogoutButton } from '@/components/LogoutButton';
import { useEffect, useRef, useState } from 'react';

export default function MT5Simulator() {
    const [config, setConfig] = useState({
        starting_balance: 10000,
        target_balance: 15000,
        overall_max_loss: 3000,
        daily_max_loss: 1000,
        time_limit_days: 7,
        weekends_tradable: false
    });
    const iframeRef = useRef<HTMLIFrameElement | null>(null);

    const syncConfigToIframe = () => {
        if (!iframeRef.current?.contentWindow) return;
        iframeRef.current.contentWindow.postMessage(
            {
                type: 'MT5_CONFIG_UPDATE',
                payload: config
            },
            window.location.origin
        );
    };

    useEffect(() => {
        syncConfigToIframe();
    }, [config]);

    return (
        <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column', background: '#0B0B0F' }}>
            <div style={{
                background: 'rgba(255, 122, 24, 0.15)',
                border: '1px solid rgba(255, 122, 24, 0.45)',
                borderRadius: '8px',
                padding: '0.8rem 1rem',
                margin: '1rem 1rem 0.5rem 1rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.9rem'
            }}>
                <span style={{ color: '#FF9A45', fontWeight: 600 }}>🔵 MT5 Personal Account Mode</span>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                    <a href="/simulator/ninjatrader" style={{
                        fontSize: '0.78rem',
                        padding: '0.3rem 0.7rem',
                        borderRadius: '6px',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        background: 'transparent',
                        color: '#666',
                        textDecoration: 'none',
                        cursor: 'pointer',
                        transition: 'all 0.15s'
                    }}>Switch to NinjaTrader</a>
                    <a href="/simulator/tradingview" style={{
                        fontSize: '0.78rem',
                        padding: '0.3rem 0.7rem',
                        borderRadius: '6px',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        background: 'transparent',
                        color: '#666',
                        textDecoration: 'none',
                        cursor: 'pointer',
                        transition: 'all 0.15s'
                    }}>Switch to TradingView</a>
                    <LogoutButton />
                </div>
            </div>

            {/* Account Configuration Panel */}
            <div style={{
                background: 'rgba(255, 122, 24, 0.08)',
                border: '1px solid rgba(255, 122, 24, 0.25)',
                borderRadius: '8px',
                padding: '1rem',
                margin: '0.5rem 1rem',
                color: '#e0e0e0'
            }}>
                <h3 style={{ fontSize: '0.85rem', color: '#FF9A45', marginBottom: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Account Configuration
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr 1fr', gap: '0.6rem' }}>
                    <div>
                        <label style={{ fontSize: '0.7rem', color: '#999', display: 'block', marginBottom: '0.3rem' }}>
                            Starting Balance ($)
                        </label>
                        <input
                            type="number"
                            value={config.starting_balance}
                            onChange={(e) => setConfig({ ...config, starting_balance: parseFloat(e.target.value) })}
                            style={{
                                width: '100%',
                                padding: '0.4rem 0.5rem',
                                background: '#0B0B0F',
                                border: '1px solid rgba(255, 122, 24, 0.3)',
                                borderRadius: '6px',
                                color: '#e0e0e0',
                                fontSize: '0.8rem'
                            }}
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '0.7rem', color: '#999', display: 'block', marginBottom: '0.3rem' }}>
                            Target Balance ($)
                        </label>
                        <input
                            type="number"
                            value={config.target_balance}
                            onChange={(e) => setConfig({ ...config, target_balance: parseFloat(e.target.value) })}
                            style={{
                                width: '100%',
                                padding: '0.4rem 0.5rem',
                                background: '#0B0B0F',
                                border: '1px solid rgba(255, 122, 24, 0.3)',
                                borderRadius: '6px',
                                color: '#e0e0e0',
                                fontSize: '0.8rem'
                            }}
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '0.7rem', color: '#999', display: 'block', marginBottom: '0.3rem' }}>
                            Overall Max Loss ($)
                        </label>
                        <input
                            type="number"
                            value={config.overall_max_loss}
                            onChange={(e) => setConfig({ ...config, overall_max_loss: parseFloat(e.target.value) })}
                            style={{
                                width: '100%',
                                padding: '0.4rem 0.5rem',
                                background: '#0B0B0F',
                                border: '1px solid rgba(255, 122, 24, 0.3)',
                                borderRadius: '6px',
                                color: '#e0e0e0',
                                fontSize: '0.8rem'
                            }}
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '0.7rem', color: '#999', display: 'block', marginBottom: '0.3rem' }}>
                            Daily Max Loss ($)
                        </label>
                        <input
                            type="number"
                            value={config.daily_max_loss}
                            onChange={(e) => setConfig({ ...config, daily_max_loss: parseFloat(e.target.value) })}
                            style={{
                                width: '100%',
                                padding: '0.4rem 0.5rem',
                                background: '#0B0B0F',
                                border: '1px solid rgba(255, 122, 24, 0.3)',
                                borderRadius: '6px',
                                color: '#e0e0e0',
                                fontSize: '0.8rem'
                            }}
                        />
                    </div>
                    <div>
                        <label style={{ fontSize: '0.7rem', color: '#999', display: 'block', marginBottom: '0.3rem' }}>
                            Time Limit (days)
                        </label>
                        <input
                            type="number"
                            value={config.time_limit_days}
                            onChange={(e) => setConfig({ ...config, time_limit_days: parseInt(e.target.value) })}
                            style={{
                                width: '100%',
                                padding: '0.4rem 0.5rem',
                                background: '#0B0B0F',
                                border: '1px solid rgba(255, 122, 24, 0.3)',
                                borderRadius: '6px',
                                color: '#e0e0e0',
                                fontSize: '0.8rem'
                            }}
                        />
                    </div>
                    <div style={{ display: 'flex', alignItems: 'flex-end' }}>
                        <label style={{ fontSize: '0.7rem', color: '#999', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                            <input
                                type="checkbox"
                                checked={config.weekends_tradable}
                                onChange={(e) => setConfig({ ...config, weekends_tradable: e.target.checked })}
                                style={{ width: '14px', height: '14px', cursor: 'pointer' }}
                            />
                            Weekends tradable
                        </label>
                    </div>
                </div>
            </div>

            <iframe
                ref={iframeRef}
                src="/simulator-ui.html?v=20260324b&uploadMode=mt5"
                onLoad={syncConfigToIframe}
                style={{ flex: 1, border: 'none', borderRadius: '8px', margin: '0.5rem 1rem 1rem 1rem' }}
            />
        </div>
    );
}
