export default function TestDashboard() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold text-gray-800 dark:text-white">
        🚀 Test Dashboard
      </h1>
      <p className="text-gray-600 dark:text-gray-300 mt-2">
        Hvis du ser denne teksten, virker React og TypeScript!
      </p>
      <div className="mt-4 p-4 bg-blue-100 dark:bg-blue-900 rounded-lg">
        <p className="text-blue-800 dark:text-blue-200">
          ✅ Frontend kjører korrekt!
        </p>
      </div>
    </div>
  );
}