import { useState, useEffect } from 'react';

function _readDark(): boolean {
  return document.body.getAttribute('data-jp-theme-light') === 'false';
}

export function useDarkMode(): boolean {
  const [isDark, setIsDark] = useState(_readDark);

  useEffect(() => {
    const observer = new MutationObserver(() => setIsDark(_readDark()));
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-jp-theme-light']
    });
    return () => observer.disconnect();
  }, []);

  return isDark;
}
