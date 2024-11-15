import {LibraryBig} from "lucide-react";

export default function KnowledgeBaseEmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-[50vh] gap-6 rounded-md">
      <div className="flex items-center justify-center w-20 h-20 rounded-full bg-gray-200 dark:bg-gray-800">
        <LibraryBig size={40}/>
      </div>
      <div className="space-y-2 text-center">
        <h2 className="text-2xl font-bold tracking-tight">No knowledge base to display</h2>
        <p className="text-gray-500 dark:text-gray-400">
          To enable AI assistant generate more accurate answers, please follow the steps:
        </p>
        <p className="text-gray-500 dark:text-gray-400">
          1. Create a knowledge base -&gt;
          2. Import the documents from certain domain -&gt;
          3. Linked the knowledge base to the chat engine
        </p>
      </div>
    </div>
  )
}