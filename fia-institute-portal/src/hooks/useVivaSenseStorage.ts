import { useState, useEffect, useCallback } from "react";
import { VivaSenseResultsData } from "@/components/vivasense/VivaSenseResults";

const STORAGE_KEY = "vivasense_last_results";

interface StoredData {
  results: VivaSenseResultsData;
  userLevel: string;
  timestamp: number;
}

export function useVivaSenseStorage() {
  const [storedResults, setStoredResults] = useState<VivaSenseResultsData | null>(null);
  const [storedUserLevel, setStoredUserLevel] = useState<string>("");

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const data: StoredData = JSON.parse(stored);
        // Check if data is less than 24 hours old
        const isValid = Date.now() - data.timestamp < 24 * 60 * 60 * 1000;
        if (isValid && data.results) {
          setStoredResults(data.results);
          setStoredUserLevel(data.userLevel || "");
        } else {
          localStorage.removeItem(STORAGE_KEY);
        }
      }
    } catch (error) {
      console.error("Error loading stored results:", error);
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const saveResults = useCallback((results: VivaSenseResultsData, userLevel: string) => {
    try {
      const data: StoredData = {
        results,
        userLevel,
        timestamp: Date.now(),
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      setStoredResults(results);
      setStoredUserLevel(userLevel);
    } catch (error) {
      console.error("Error saving results:", error);
    }
  }, []);

  const clearResults = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setStoredResults(null);
    setStoredUserLevel("");
  }, []);

  return {
    storedResults,
    storedUserLevel,
    saveResults,
    clearResults,
  };
}
