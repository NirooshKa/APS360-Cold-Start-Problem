using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ResampleDataset
{
    class Program
    {
        static void Main(string[] args)
        {
            if(args.Length == 4)
            {
                ResampleDataset(args[0], args[1], Convert.ToDouble(args[2]), args[3].ToLower().Trim() == "copy");
            }
            else if(args.Length == 0)
            {
                Console.Write("Source Dataset Folder: ");
                string source = Console.ReadLine();
                Console.Write("Target Dataset Folder: ");
                string target = Console.ReadLine();
                Console.Write("Percentage to Keep: ");
                double percentage = Convert.ToDouble(Console.ReadLine());
                Console.Write("Do you want to move or copy? (Type \"Move\"/\"Copy\") ");
                bool copyInsteadOfMove = Console.ReadLine().ToLower().Trim() == "copy";
                ResampleDataset(source, target, percentage, copyInsteadOfMove);
            }
            else
            {
                Console.WriteLine("Invalid number of arguments. Require 4 arguments: SourceFolder:String, TargetFolder:String, Percentage:Double, CopyInsteadOfMove:Bool.");
            }
        }

        private static void ResampleDataset(string source, string target, double percentage, bool copyInsteadOfMove = false)
        {
            Random random = new Random();

            // Input validation
            if (!Directory.Exists(source))
            {
                Console.WriteLine("Source folder doesn't exist!");
                return;
            }
            if(Directory.Exists(target))
            {
                Console.WriteLine("Target folder already exist!");
                return;
            }
            if(percentage < 0 || percentage > 1)
            {
                Console.WriteLine("Range for percentage is invalid.");
            }

            // Create target parent folder
            Directory.CreateDirectory(target);
            // Iterate through and move source
            foreach (var classFolder in Directory.EnumerateDirectories(source))
            {
                string className = Path.GetFileName(classFolder);
                string targetClassFolder = Path.Combine(target, className);

                Console.WriteLine($"Moving `{className}`...");

                // Random sample a bunch of cats
                var total = Directory.EnumerateFiles(classFolder).ToArray();
                List<string> files = total.Where(f => random.NextDouble() < percentage).ToList();
                // Validation target size
                if(files.Count == 0)
                {
                    Console.WriteLine($"Sample size too small ({files.Count}/{total.Length}), skip.");
                    continue;
                }
                // Create target folder
                Directory.CreateDirectory(targetClassFolder);
                // Move the file
                files.ForEach(f =>
                {
                    string fileName = Path.GetFileName(f);
                    string targetFileName = Path.Combine(targetClassFolder, fileName);
                    if (copyInsteadOfMove)
                        File.Copy(f, targetFileName);
                    else
                        File.Move(f, targetFileName);
                });

                Console.WriteLine($"{(copyInsteadOfMove ? "Copied" : "Moved")} {files.Count}/{total.Length} files.");
            }
        }
    }
}
