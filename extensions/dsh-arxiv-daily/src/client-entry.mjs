import { installClient } from './client.mjs';
window.__ModuleLoader__.load({
  id: 'dsh-arxiv-daily',
  factory(require) {
    const React = require('react');
    return { inject: ['slots', 'locale', 'connection', 'sidebarRightTabs'], apply(ctx) { installClient(ctx, React, window.location); } };
  },
});
