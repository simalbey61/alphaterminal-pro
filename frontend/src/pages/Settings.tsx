import React from 'react';
import { Settings as SettingsIcon, Bell, Shield, Palette, Globe, Key } from 'lucide-react';

export default function Settings() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Ayarlar</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* General */}
          <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
            <div className="flex items-center space-x-3 mb-6">
              <SettingsIcon className="w-5 h-5 text-primary-500" />
              <h2 className="font-semibold text-white">Genel</h2>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white">Dil</p>
                  <p className="text-sm text-surface-500">Arayüz dili</p>
                </div>
                <select className="bg-surface-800 border border-surface-700 rounded-lg px-3 py-2">
                  <option>Türkçe</option>
                  <option>English</option>
                </select>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white">Tema</p>
                  <p className="text-sm text-surface-500">Görünüm tercihi</p>
                </div>
                <select className="bg-surface-800 border border-surface-700 rounded-lg px-3 py-2">
                  <option>Koyu</option>
                  <option>Açık</option>
                </select>
              </div>
            </div>
          </div>

          {/* Notifications */}
          <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
            <div className="flex items-center space-x-3 mb-6">
              <Bell className="w-5 h-5 text-yellow-500" />
              <h2 className="font-semibold text-white">Bildirimler</h2>
            </div>
            <div className="space-y-4">
              {['Sinyal Bildirimleri', 'Trade Bildirimleri', 'Risk Uyarıları', 'Günlük Rapor'].map((item) => (
                <div key={item} className="flex items-center justify-between">
                  <span className="text-white">{item}</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" defaultChecked className="sr-only peer" />
                    <div className="w-11 h-6 bg-surface-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
                  </label>
                </div>
              ))}
            </div>
          </div>

          {/* API Keys */}
          <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
            <div className="flex items-center space-x-3 mb-6">
              <Key className="w-5 h-5 text-green-500" />
              <h2 className="font-semibold text-white">API Anahtarları</h2>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-surface-400 mb-2">Telegram Bot Token</label>
                <input type="password" placeholder="••••••••••••" className="w-full bg-surface-800 border border-surface-700 rounded-lg px-3 py-2" />
              </div>
              <div>
                <label className="block text-sm text-surface-400 mb-2">Telegram Chat ID</label>
                <input type="text" placeholder="Chat ID" className="w-full bg-surface-800 border border-surface-700 rounded-lg px-3 py-2" />
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
            <h2 className="font-semibold text-white mb-4">Sürüm Bilgisi</h2>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-surface-400">Sürüm</span>
                <span className="text-white">v4.2.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-surface-400">Build</span>
                <span className="text-white">2024.01.09</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
